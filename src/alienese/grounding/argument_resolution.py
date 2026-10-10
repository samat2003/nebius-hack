"""Grounded argument resolver for external tool bindings.

Binds extracted grounding evidence to tool schema parameters without fabricating
arguments, guessing property names, or coercing types unsafely.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from alienese.contracts.state import (
    CanonicalCapability,
    ExternalToolBinding,
    WorkingState,
)
from alienese.engine.turn import (
    _validate_schema_node,
    _validate_schema_structure,
    validate_tool_arguments_against_schema,
)
from alienese.grounding.evidence import EvidenceCategory, GroundingEvidence

_PATH_PROPERTY_NAMES: tuple[str, ...] = (
    "path",
    "filepath",
    "file_path",
    "target_file",
    "filename",
    "file",
)

_TEST_TARGET_PROPERTY_NAMES: tuple[str, ...] = (
    "target",
    "test_target",
    "node_id",
    "test_path",
    "test",
    "tests",
    "path",
)

_SEARCH_QUERY_PROPERTY_NAMES: tuple[str, ...] = (
    "query",
    "pattern",
    "text",
    "search_term",
    "substring",
)

_COMMAND_PROPERTY_NAMES: tuple[str, ...] = (
    "command",
    "cmd",
    "command_line",
)


class GroundedArgumentResolver:
    """Safely grounds tool arguments against tool schemas and extracted evidence."""

    @staticmethod
    def extract_defaults(schema: Mapping[str, Any]) -> dict[str, Any]:
        """Extract schema-defined defaults for properties."""
        defaults: dict[str, Any] = {}
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for prop_name, prop_def in properties.items():
                if isinstance(prop_def, Mapping) and "default" in prop_def:
                    default_val = copy.deepcopy(prop_def["default"])
                    ok, _ = _validate_schema_node(
                        prop_def,
                        default_val,
                        path=str(prop_name),
                        depth=1,
                    )
                    if ok:
                        defaults[str(prop_name)] = default_val
        return defaults

    def resolve_arguments(
        self,
        binding: ExternalToolBinding,
        evidence_items: Sequence[GroundingEvidence],
        state: WorkingState,
        primary_evidence: GroundingEvidence | None = None,
    ) -> tuple[dict[str, Any], bool, tuple[GroundingEvidence, ...]]:
        """Resolve complete arguments for a tool binding from evidence.

        Returns:
            (arguments_dict, is_complete, bound_evidence_tuple)
        """
        schema = binding.parameters_schema
        if not isinstance(schema, Mapping):
            return {}, False, ()

        struct_ok, _ = _validate_schema_structure(schema, path="$", depth=0)
        if not struct_ok:
            return {}, False, ()

        if schema.get("type", "object") != "object":
            return {}, False, ()

        arguments = self.extract_defaults(schema)
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}

        raw_required = schema.get("required", [])
        required_props = set(raw_required) if isinstance(raw_required, Sequence) else set()

        bound_evidence: list[GroundingEvidence] = []
        capability = binding.canonical_capability

        # Attempt capability-directed argument mapping
        if capability in (
            CanonicalCapability.READ_FILE,
            CanonicalCapability.LIST_FILES,
            CanonicalCapability.WRITE_FILE,
            CanonicalCapability.APPLY_PATCH,
        ):
            self._ground_path_argument(
                properties,
                arguments,
                evidence_items,
                primary_evidence,
                bound_evidence,
            )

        elif capability == CanonicalCapability.RUN_TEST:
            self._ground_test_argument(
                properties,
                arguments,
                evidence_items,
                primary_evidence,
                bound_evidence,
            )

        elif capability in (
            CanonicalCapability.SEARCH_TEXT,
            CanonicalCapability.SYNTHESIZE_SEARCH,
        ):
            self._ground_search_argument(
                properties,
                arguments,
                evidence_items,
                primary_evidence,
                bound_evidence,
            )

        elif capability == CanonicalCapability.RUN_COMMAND:
            self._ground_command_argument(
                properties,
                arguments,
                evidence_items,
                primary_evidence,
                bound_evidence,
            )

        # Validate arguments against the fail-closed JSON Schema
        is_valid, _reason = validate_tool_arguments_against_schema(schema, arguments)
        is_complete = is_valid and required_props.issubset(arguments.keys())

        # High risk mutations (APPLY_PATCH, WRITE_FILE) must not be executable
        # if patch or content was fabricated or missing
        if capability in (CanonicalCapability.APPLY_PATCH, CanonicalCapability.WRITE_FILE):
            for content_key in ("patch", "content", "code", "file_text", "new_content"):
                if content_key in properties and content_key not in arguments:
                    is_complete = False

        return arguments, is_complete, tuple(bound_evidence)

    def _ground_path_argument(
        self,
        properties: Mapping[str, Any],
        arguments: dict[str, Any],
        evidence_items: Sequence[GroundingEvidence],
        primary_evidence: GroundingEvidence | None,
        bound_evidence: list[GroundingEvidence],
    ) -> None:
        target_prop = None
        for prop in _PATH_PROPERTY_NAMES:
            if prop in properties:
                target_prop = prop
                break
        if not target_prop or target_prop in arguments:
            return

        # Prefer primary evidence if FILE_PATH
        if primary_evidence and primary_evidence.category == EvidenceCategory.FILE_PATH:
            arguments[target_prop] = primary_evidence.value
            bound_evidence.append(primary_evidence)
            return

        # Search extracted evidence
        for evi in evidence_items:
            if evi.category == EvidenceCategory.FILE_PATH:
                arguments[target_prop] = evi.value
                bound_evidence.append(evi)
                return

    def _ground_test_argument(
        self,
        properties: Mapping[str, Any],
        arguments: dict[str, Any],
        evidence_items: Sequence[GroundingEvidence],
        primary_evidence: GroundingEvidence | None,
        bound_evidence: list[GroundingEvidence],
    ) -> None:
        target_prop = None
        for prop in _TEST_TARGET_PROPERTY_NAMES:
            if prop in properties:
                target_prop = prop
                break
        if not target_prop or target_prop in arguments:
            return

        # Prefer primary evidence if TEST_TARGET
        if primary_evidence and primary_evidence.category == EvidenceCategory.TEST_TARGET:
            arguments[target_prop] = primary_evidence.value
            bound_evidence.append(primary_evidence)
            return

        # Search extracted evidence for TEST_TARGET
        for evi in evidence_items:
            if evi.category == EvidenceCategory.TEST_TARGET:
                arguments[target_prop] = evi.value
                bound_evidence.append(evi)
                return

        # Fallback to test file paths
        for evi in evidence_items:
            if evi.category == EvidenceCategory.FILE_PATH and ("test" in evi.value.lower()):
                arguments[target_prop] = evi.value
                bound_evidence.append(evi)
                return

    def _ground_search_argument(
        self,
        properties: Mapping[str, Any],
        arguments: dict[str, Any],
        evidence_items: Sequence[GroundingEvidence],
        primary_evidence: GroundingEvidence | None,
        bound_evidence: list[GroundingEvidence],
    ) -> None:
        target_prop = None
        for prop in _SEARCH_QUERY_PROPERTY_NAMES:
            if prop in properties:
                target_prop = prop
                break
        if not target_prop or target_prop in arguments:
            return

        if primary_evidence and primary_evidence.category in (
            EvidenceCategory.SEARCH_PATTERN,
            EvidenceCategory.SYMBOL,
        ):
            arguments[target_prop] = primary_evidence.value
            bound_evidence.append(primary_evidence)
            return

        for evi in evidence_items:
            if evi.category in (EvidenceCategory.SEARCH_PATTERN, EvidenceCategory.SYMBOL):
                arguments[target_prop] = evi.value
                bound_evidence.append(evi)
                return

    def _ground_command_argument(
        self,
        properties: Mapping[str, Any],
        arguments: dict[str, Any],
        evidence_items: Sequence[GroundingEvidence],
        primary_evidence: GroundingEvidence | None,
        bound_evidence: list[GroundingEvidence],
    ) -> None:
        target_prop = None
        for prop in _COMMAND_PROPERTY_NAMES:
            if prop in properties:
                target_prop = prop
                break
        if not target_prop or target_prop in arguments:
            return

        # Only ground exact observed TEST_COMMAND
        if primary_evidence and primary_evidence.category == EvidenceCategory.TEST_COMMAND:
            arguments[target_prop] = primary_evidence.value
            bound_evidence.append(primary_evidence)
            return

        for evi in evidence_items:
            if evi.category == EvidenceCategory.TEST_COMMAND:
                arguments[target_prop] = evi.value
                bound_evidence.append(evi)
                return
