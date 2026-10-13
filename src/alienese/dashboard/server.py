"""Alienese Live Visual Companion Dashboard Server.

Serves an authentic real-time visual dashboard at http://127.0.0.1:4097
displaying three-model orchestration, scientifically valid context reduction metrics,
EmbeddingGemma dense rankings, mini-Jev decision policy scores, plan adherence,
and RED -> GREEN verification obligations strictly scoped to active sessions.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger("alienese.dashboard")

DEFAULT_PORT = 4097
DEFAULT_WORKSPACE = Path("/tmp/alienese_opencode_demo")


def check_service_health(
    base_url: str = "http://127.0.0.1:8001",
) -> dict[str, dict[str, str]]:
    """Check live model server readiness and Nebius key configuration."""
    retriever_status = "Unavailable"
    controller_status = "Unavailable"
    retriever_detail = "Offline"
    controller_detail = "Offline"
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=0.25) as resp:
            if resp.status == 200:
                hdata = json.loads(resp.read().decode("utf-8"))
                if hdata.get("status") == "ok":
                    r_info = hdata.get("retriever", {})
                    c_info = hdata.get("controller", {})
                    if r_info.get("status") == "ready":
                        retriever_status = "Ready"
                        retriever_detail = f"Online ({r_info.get('device', 'cpu')})"
                    if c_info.get("status") == "ready":
                        controller_status = "Ready"
                        controller_detail = f"Online ({c_info.get('device', 'cpu')})"
    except Exception:
        pass

    gen_status = "Unavailable"
    gen_detail = "Not configured"
    has_nebius = bool(os.getenv("NEBIUS_TOKEN_FACTORY_KEY") or os.getenv("NEBIUS_API_KEY"))
    if not has_nebius:
        for env_path in [Path(".env"), Path(__file__).resolve().parents[3] / ".env"]:
            if env_path.is_file():
                try:
                    txt = env_path.read_text(encoding="utf-8")
                    if "NEBIUS_TOKEN_FACTORY_KEY" in txt or "NEBIUS_API_KEY" in txt:
                        has_nebius = True
                        break
                except Exception:
                    pass
    if has_nebius:
        gen_status = "Ready"
        gen_detail = "API key configured"

    return {
        "retriever": {"status": retriever_status, "detail": retriever_detail},
        "controller": {"status": controller_status, "detail": controller_detail},
        "generator": {"status": gen_status, "detail": gen_detail},
    }


def read_workspace_data(
    workspace: Path, target_session_id: str | None = None
) -> dict[str, Any]:
    """Parse real audit trace and session records for an explicitly selected session."""
    ws = workspace.resolve()
    trace_file = ws / ".alienese" / "opencode_trace.jsonl"
    session_dir = ws / ".alienese" / "sessions"

    trace_records: list[dict[str, Any]] = []
    if trace_file.is_file():
        try:
            with open(trace_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trace_records.append(json.loads(line))
        except Exception as e:
            logger.warning("Error reading trace file: %s", e)

    # Discover all session IDs from session directory and trace records
    discovered_sessions: dict[str, float] = {}
    if session_dir.is_dir():
        for p in session_dir.glob("*.json"):
            sid = p.stem
            discovered_sessions[sid] = p.stat().st_mtime
    for r in trace_records:
        r_sid = str(r.get("session_id") or "")
        if r_sid and r_sid not in discovered_sessions:
            discovered_sessions[r_sid] = 0.0

    available_sessions = sorted(
        discovered_sessions.keys(),
        key=lambda s: discovered_sessions.get(s, 0.0),
        reverse=True,
    )

    # Select active session ID
    if target_session_id and (target_session_id in discovered_sessions or trace_records):
        active_session_id = target_session_id
    elif available_sessions:
        active_session_id = available_sessions[0]
    elif trace_records:
        active_session_id = str(trace_records[-1].get("session_id") or "default")
    else:
        active_session_id = "default"

    # Load session state for the active session
    session_data: dict[str, Any] = {}
    session_file = session_dir / f"{active_session_id}.json"
    if session_file.is_file():
        try:
            session_data = json.loads(session_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Error reading session file %s: %s", session_file, e)

    # CRITICAL: Filter trace records strictly to the active session
    session_trace_records = [
        r for r in trace_records if r.get("session_id") == active_session_id
    ]

    # Extract latest plan strictly from this session's trace
    plan_entry = next(
        (r for r in reversed(session_trace_records) if r.get("phase") == "ALIENESE_PLAN"),
        None,
    )
    plan_details = plan_entry.get("details", {}) if plan_entry else {}

    # Context reduction metrics: candidate context reduction (Metric A)
    # NO FABRICATED NUMBERS! If not measured, candidate_context_reduction is None.
    candidate_reduction = (
        plan_details.get("candidate_context_reduction")
        or plan_details.get("context_reduction")
    )
    # Actual API token savings (Metric B)
    actual_api_savings = plan_details.get("actual_api_savings")

    # Extract EmbeddingGemma rankings strictly from this session's plan
    ranked_files = plan_details.get("ranked_files", [])

    # Extract mini-Jev options and scores strictly from this session's plan
    jev_options = plan_details.get("options", [])
    selected_id = plan_details.get("selected_id")
    selected_label = plan_details.get("selected_label")
    selected_prob = plan_details.get("selected_probability", 0.0)

    # Extract plan adherence strictly from this session
    adherence = session_data.get("plan_adherence") or session_data.get("last_adherence")
    if not adherence:
        for r in session_trace_records:
            adh = r.get("adherence") or (r.get("details") or {}).get("adherence")
            if adh:
                adherence = adh
                break

    # Extract mutations strictly from this session
    mutations: list[str] = list(session_data.get("mutations", []))
    for r in session_trace_records:
        for m in r.get("mutations", []):
            if m not in mutations:
                mutations.append(m)

    verification_obligation = bool(session_data.get("verification_obligation", False))
    last_test_passed = session_data.get("last_test_passed")

    # Derive verification lifecycle status strictly for this session
    if last_test_passed is True and len(mutations) > 0 and not verification_obligation:
        verification_status = "GREEN_VERIFIED"
        verification_status_label = "VERIFIED (GREEN)"
        verification_status_color = "#10B981"
    elif verification_obligation:
        verification_status = "MUTATION_ACTIVE"
        verification_status_label = "MUTATION ACTIVE (TEST REQUIRED)"
        verification_status_color = "#F59E0B"
    elif last_test_passed is False:
        verification_status = "FAILING_REPRODUCED"
        verification_status_label = "REPRODUCED (RED)"
        verification_status_color = "#EF4444"
    else:
        verification_status = "INITIALIZING"
        verification_status_label = "STANDBY"
        verification_status_color = "#64748B"

    # Query live service health
    svc_health = check_service_health()

    return {
        "workspace": str(ws),
        "session_id": active_session_id,
        "available_sessions": available_sessions,
        "step_count": session_data.get("sequence_no", len(session_trace_records)),
        "state_digest": session_data.get("state_digest", "0" * 64)[:16],
        "state_digest_full": session_data.get("state_digest", "0" * 64),
        "models": {
            "retriever": {
                "name": "EmbeddingGemma 2",
                "role": "Dense Matryoshka Context Ranking (768d)",
                "status": svc_health["retriever"]["status"],
                "detail": svc_health["retriever"]["detail"],
            },
            "controller": {
                "name": "mini-Jev",
                "revision": "step-010626",
                "role": "Bounded Next-Action Policy Head (Qwen3-0.6B)",
                "status": svc_health["controller"]["status"],
                "detail": svc_health["controller"]["detail"],
            },
            "generator": {
                "name": "NVIDIA Nemotron 3 Super (120B-A12B)",
                "provider": "Nebius Token Factory",
                "role": "Reasoning & Targeted Code Synthesis",
                "status": svc_health["generator"]["status"],
                "detail": svc_health["generator"]["detail"],
            },
        },
        "candidate_context_reduction": candidate_reduction,
        "context_reduction": candidate_reduction,
        "actual_api_savings": actual_api_savings,
        "ranked_files": ranked_files,
        "jev_options": jev_options,
        "selected_id": selected_id,
        "selected_label": selected_label,
        "selected_probability": selected_prob,
        "adherence": adherence,
        "mutations": mutations,
        "verification_obligation": verification_obligation,
        "last_test_passed": last_test_passed,
        "verification_status": verification_status,
        "verification_status_label": verification_status_label,
        "verification_status_color": verification_status_color,
        "tokens_per_model": None,
        "trace_events": session_trace_records[-20:],
        "trace_total_count": len(session_trace_records),
    }


def render_svg_bar(
    label: str, value: float, max_value: float, fill_color: str, is_highlighted: bool = False
) -> str:
    """Render a clean SVG bar row for metric charts."""
    pct = max(0.0, min(100.0, (value / max(max_value, 0.001)) * 100.0))
    esc_label = html.escape(label)
    highlight_style = "stroke:#10B981;stroke-width:2;" if is_highlighted else ""
    badge = ' <tspan fill="#10B981" font-weight="bold">★ SELECTED</tspan>' if is_highlighted else ""

    bar_head = (
        "display: flex; justify-content: space-between; font-size: 13px; "
        "font-family: monospace; color: #E2E8F0; margin-bottom: 4px;"
    )
    bar_track = (
        "background: rgba(255,255,255,0.06); border-radius: 4px; "
        "height: 10px; overflow: hidden; position: relative;"
    )
    bar_fill = (
        f"background: {fill_color}; width: {pct}%; height: 100%; "
        f"border-radius: 4px; transition: width 0.4s ease; {highlight_style}"
    )
    return f"""
    <div style="margin-bottom: 8px;">
      <div style="{bar_head}">
        <span>{esc_label} {badge}</span>
        <span style="font-weight: 600; color: #94A3B8;">{value:.4f} ({pct:.1f}%)</span>
      </div>
      <div style="{bar_track}">
        <div style="{bar_fill}"></div>
      </div>
    </div>
    """


def generate_html_dashboard(data: dict[str, Any]) -> str:
    """Generate the complete modern dark UI HTML dashboard page."""
    # Metric A: Candidate Context Reduction
    cr = data.get("candidate_context_reduction")
    if cr and cr.get("reduction_pct") is not None:
        reduction_pct_str = f"{cr['reduction_pct']}%"
        sel_tok = cr.get("selected_tokens", 0)
        tot_tok = cr.get("total_candidate_tokens", 0)
        tok_method = cr.get("tokenizer", "tiktoken (cl100k_base)")
        cr_sub = f"{sel_tok:,} selected / {tot_tok:,} candidate tokens"
        cr_detail = f"Tokenizer: {tok_method}"
        cr_color = "#38BDF8"
    else:
        reduction_pct_str = "Not measured"
        cr_sub = "Awaiting alienese_plan execution"
        cr_detail = "Formula: 1 - (selected / candidate)"
        cr_color = "#64748B"

    # Metric B: Actual LLM API Token Savings
    api_sav = data.get("actual_api_savings")
    if api_sav and api_sav.get("savings_pct") is not None:
        api_sav_val = f"{api_sav['savings_pct']}%"
        a_tok = api_sav.get("alienese_tokens", 0)
        b_tok = api_sav.get("baseline_tokens", 0)
        api_sav_sub = f"{a_tok:,} tokens vs {b_tok:,} baseline"
        api_sav_src = str(api_sav.get("source", "Nebius API"))
        api_sav_color = "#10B981"
    else:
        api_sav_val = "Not measured"
        api_sav_sub = "Requires baseline benchmark run"
        api_sav_src = "No synthetic estimate"
        api_sav_color = "#64748B"

    # mini-Jev Decision Card
    if data.get("selected_id"):
        jev_title = html.escape(str(data.get("selected_label") or "Standing By"))
        jev_score = f"{data.get('selected_probability', 0.0):.1%}"
        jev_id = html.escape(str(data.get("selected_id")))
        jev_color = "#A855F7"
    else:
        jev_title = "Standing By"
        jev_score = "Not measured"
        jev_id = "none"
        jev_color = "#64748B"

    # EmbeddingGemma Ranking bars
    gemma_bars_html = ""
    ranked_files = data.get("ranked_files", [])
    max_score = max([s for _, s in ranked_files], default=1.0)
    if ranked_files:
        for idx, (fpath, score) in enumerate(ranked_files):
            fill = "#38BDF8" if idx == 0 else "#64748B"
            gemma_bars_html += render_svg_bar(fpath, score, max_score, fill, idx == 0)
    else:
        gemma_bars_html = (
            f'<div style="color: #64748B; font-style: italic; padding: 12px 0;">'
            f'No retrieval rankings recorded for session {data["session_id"]}. '
            f'Run alienese_plan.</div>'
        )

    # mini-Jev Options bars
    jev_bars_html = ""
    jev_options = data.get("jev_options", [])
    sel_id = data.get("selected_id")
    if jev_options:
        for opt in jev_options:
            is_sel = opt.get("id") == sel_id
            fill = "#10B981" if is_sel else "#A855F7"
            prob = opt.get("probability", 0.0)
            lbl = f"{opt.get('id')}: {opt.get('label', opt.get('id'))}"
            jev_bars_html += render_svg_bar(lbl, prob, 1.0, fill, is_sel)
    else:
        jev_bars_html = (
            f'<div style="color: #64748B; font-style: italic; padding: 12px 0;">'
            f'No controller choices recorded for session {data["session_id"]}.</div>'
        )

    # Adherence display
    adh = data.get("adherence")
    if adh:
        is_matched = adh.get("matched", False)
        badge_style = "padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 12px;"
        adh_badge = (
            f'<span style="background: rgba(16,185,129,0.15); color: #10B981; '
            f'border: 1px solid #10B981; {badge_style}">MATCHED</span>'
            if is_matched
            else f'<span style="background: rgba(245,158,11,0.15); color: #F59E0B; '
            f'border: 1px solid #F59E0B; {badge_style}">DEVIATED</span>'
        )
        rec_str = html.escape(str(adh.get("recommended")))
        act_str = html.escape(str(adh.get("actual")))
        adh_details = f"""
        <div style="margin-top: 10px; font-family: monospace; font-size: 13px;">
          <div style="color: #94A3B8;">Recommended:
            <span style="color: #E2E8F0;">{rec_str}</span>
          </div>
          <div style="color: #94A3B8; margin-top: 4px;">1st Action:
            <span style="color: #38BDF8;">{act_str}</span>
          </div>
        </div>
        """
    else:
        badge_style = "padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 12px;"
        adh_badge = (
            f'<span style="background: rgba(100,116,139,0.15); color: #94A3B8; '
            f'border: 1px solid #64748B; {badge_style}">PENDING PLAN</span>'
        )
        adh_details = (
            '<div style="margin-top: 10px; color: #64748B; font-size: 13px; font-style: italic;">'
            'Awaiting initial plan decision.</div>'
        )

    # Trace log entries
    trace_rows_html = ""
    for ev in reversed(data.get("trace_events", [])):
        ts = ev.get("timestamp", "").split("T")[-1][:8]
        phase = ev.get("phase", "")
        tool = ev.get("tool", "")
        digest = (ev.get("state_digest", "") or "")[:12]
        badge_color = (
            "#38BDF8" if phase == "ALIENESE_PLAN" else ("#10B981" if "POST" in phase else "#F59E0B")
        )
        args_short = html.escape(json.dumps(ev.get("arguments", {}))[:80])
        trace_rows_html += f"""
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 12px;">
          <td style="padding: 8px 12px; color: #64748B;">{ts}</td>
          <td style="padding: 8px 12px;">
            <span style="color: {badge_color}; font-weight: 600;">{phase}</span>
          </td>
          <td style="padding: 8px 12px; color: #F1F5F9;">{tool}</td>
          <td style="padding: 8px 12px; color: #94A3B8;">{args_short}</td>
          <td style="padding: 8px 12px; color: #64748B;">{digest}…</td>
        </tr>
        """

    # Timeline stage classes
    v_status = data.get("verification_status")
    step1_done = True
    step2_done = len(ranked_files) > 0
    step3_done = len(jev_options) > 0
    step4_done = len(data.get("mutations", [])) > 0
    step5_done = v_status == "GREEN_VERIFIED"

    def stage_circle(done: bool, active: bool, color: str) -> str:
        r_c, g_c, b_c = color[1:3], color[3:5], color[5:7]
        bg = f"rgba({r_c},{g_c},{b_c},0.2)" if done else "rgba(255,255,255,0.05)"
        border = color if done else "rgba(255,255,255,0.15)"
        txt_color = "#FFFFFF" if done else "#64748B"
        return f"background: {bg}; border: 2px solid {border}; color: {txt_color};"

    # Model readiness badges and colors
    models = data["models"]
    r_stat = models["retriever"]["status"]
    r_col = "#10B981" if r_stat == "Ready" else "#EF4444"
    c_stat = models["controller"]["status"]
    c_col = "#10B981" if c_stat == "Ready" else "#EF4444"
    g_stat = models["generator"]["status"]
    g_col = "#10B981" if g_stat == "Ready" else "#64748B"

    # Session switcher options HTML
    avail_sessions = data.get("available_sessions", [])
    ws_esc = html.escape(str(data.get("workspace", "")))
    session_options_html = ""
    for s in avail_sessions:
        is_sel = "selected" if s == data["session_id"] else ""
        session_options_html += f'<option value="{s}" {is_sel}>{s}</option>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Alienese — Real-Time Model Orchestration Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0B1120;
      color: #F8FAFC;
      font-family: system-ui, -apple-system, sans-serif;
      padding: 24px;
      line-height: 1.5;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      margin-bottom: 24px;
    }}
    .title-group h1 {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .pulse-dot {{
      width: 10px;
      height: 10px;
      background: #10B981;
      border-radius: 50%;
      box-shadow: 0 0 12px #10B981;
    }}
    .subtitle {{
      color: #94A3B8;
      font-size: 13px;
      margin-top: 4px;
    }}
    .model-pills {{
      display: flex;
      gap: 12px;
    }}
    .model-pill {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .grid-5 {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}
    @media (max-width: 1200px) {{
      .grid-5 {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 20px;
    }}
    .card-title {{
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #94A3B8;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .card-big {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -1px;
      margin-bottom: 6px;
    }}
    .card-sub {{
      font-size: 12px;
      color: #94A3B8;
    }}
    .timeline {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: relative;
      margin: 16px 0 8px 0;
    }}
    .timeline::before {{
      content: "";
      position: absolute;
      top: 18px;
      left: 30px;
      right: 30px;
      height: 2px;
      background: rgba(255, 255, 255, 0.1);
      z-index: 1;
    }}
    .timeline-step {{
      display: flex;
      flex-direction: column;
      align-items: center;
      position: relative;
      z-index: 2;
    }}
    .timeline-node {{
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 8px;
    }}
    .timeline-label {{
      font-size: 11px;
      color: #94A3B8;
      text-align: center;
      max-width: 110px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }}
    th {{
      padding: 10px 12px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #64748B;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title-group">
      <h1><span class="pulse-dot"></span> Alienese Live Orchestration Engine</h1>
      <div class="subtitle">Multi-Model Gateway & Deterministic Policy Runtime • Port 4097</div>
      <div style="font-size: 12px; color: #94A3B8; margin-top: 6px; display: flex; gap: 8px;">
        <span>Active Session:</span>
        <select onchange="window.location.search = '?session_id=' + this.value"
          style="background: #1E293B; color: #38BDF8; border: 1px solid #475569; padding: 2px 6px;">
          {session_options_html}
        </select>
        <span>• Workspace:
          <span style="color: #E2E8F0; font-family: monospace;">{ws_esc}</span>
        </span>
        <span>• Events: <span style="color: #F8FAFC;">{data["trace_total_count"]}</span></span>
      </div>
    </div>
    <div class="model-pills">
      <div class="model-pill">
        <span>Retriever (EmbeddingGemma 2)</span>
        <span style="color: {r_col}; font-weight: 600;">{r_stat}</span>
      </div>
      <div class="model-pill">
        <span>Controller (mini-Jev)</span>
        <span style="color: {c_col}; font-weight: 600;">{c_stat}</span>
      </div>
      <div class="model-pill">
        <span>Generator (Nemotron 3 Super)</span>
        <span style="color: {g_col}; font-weight: 600;">{g_stat}</span>
      </div>
    </div>
  </div>

  <!-- Key Metrics Row -->
  <div class="grid-5">
    <!-- Candidate Context Pruning Card (Metric A) -->
    <div class="card">
      <div class="card-title">
        <span>Candidate Pruning</span>
        <span style="color: {cr_color}; font-size: 11px;">METRIC A</span>
      </div>
      <div class="card-big" style="color: {cr_color}; font-size: 24px;">{reduction_pct_str}</div>
      <div class="card-sub">{cr_sub}</div>
      <div class="card-sub" style="font-size: 11px; color: #64748B; margin-top: 2px;">
        {cr_detail}
      </div>
    </div>

    <!-- Actual LLM API Token Savings Card (Metric B) -->
    <div class="card">
      <div class="card-title">
        <span>API Token Savings</span>
        <span style="color: {api_sav_color}; font-size: 11px;">METRIC B</span>
      </div>
      <div class="card-big" style="color: {api_sav_color}; font-size: 24px;">{api_sav_val}</div>
      <div class="card-sub">{api_sav_sub}</div>
      <div class="card-sub" style="font-size: 11px; color: #64748B; margin-top: 2px;">
        Source: {api_sav_src}
      </div>
    </div>

    <!-- mini-Jev Decision Card -->
    <div class="card">
      <div class="card-title">
        <span>mini-Jev Choice</span>
        <span style="color: #A855F7;">Set-Head</span>
      </div>
      <div class="card-big" style="font-size: 17px; color: {jev_color};">
        {jev_title}
      </div>
      <div class="card-sub">
        Softmax: <span style="color: #10B981; font-weight: 600;">{jev_score}</span>
      </div>
      <div class="card-sub">ID: {jev_id}</div>
    </div>

    <!-- Plan Adherence Card -->
    <div class="card">
      <div class="card-title">
        <span>Plan Adherence</span>
        {adh_badge}
      </div>
      {adh_details}
    </div>

    <!-- Verification Status Card -->
    <div class="card">
      <div class="card-title">
        <span>Verification State</span>
        <span style="color: {data["verification_status_color"]}; font-weight: 700;">
          {data["verification_status_label"]}
        </span>
      </div>
      <div class="card-big" style="font-size: 17px; color: {data["verification_status_color"]};">
        {f"MUTATIONS: {len(data['mutations'])}" if data.get("mutations") else "CLEAN WORKTREE"}
      </div>
      <div class="card-sub">
        Digest: <span style="color: #94A3B8;">{data["state_digest"]}…</span>
      </div>
      <div class="card-sub">Events: {data["step_count"]}</div>
    </div>
  </div>

  <!-- State Machine Progression Timeline -->
  <div class="card" style="margin-bottom: 24px;">
    <div class="card-title">
      <span>RED ➔ GREEN Verification State Machine</span>
      <span style="color: #64748B;">Obligation Lifecycle</span>
    </div>
    <div class="timeline">
      <div class="timeline-step">
        <div class="timeline-node" style="{stage_circle(step1_done, False, "#EF4444")}">1</div>
        <div class="timeline-label">Failing Bug (RED)</div>
      </div>
      <div class="timeline-step">
        <div class="timeline-node" style="{stage_circle(step2_done, False, "#38BDF8")}">2</div>
        <div class="timeline-label">Dense Retrieval (Gemma)</div>
      </div>
      <div class="timeline-step">
        <div class="timeline-node" style="{stage_circle(step3_done, False, "#A855F7")}">3</div>
        <div class="timeline-label">Policy Choice (mini-Jev)</div>
      </div>
      <div class="timeline-step">
        <div class="timeline-node" style="{stage_circle(step4_done, False, "#F59E0B")}">4</div>
        <div class="timeline-label">Mutation Obligation</div>
      </div>
      <div class="timeline-step">
        <div class="timeline-node" style="{stage_circle(step5_done, False, "#10B981")}">5</div>
        <div class="timeline-label">Verified (GREEN)</div>
      </div>
    </div>
  </div>

  <!-- Charts Row: Retrieval Ranking vs Controller Policy -->
  <div class="grid-2">
    <!-- EmbeddingGemma Chart -->
    <div class="card">
      <div class="card-title">
        <span>EmbeddingGemma 2 Semantic Retrieval Ranking</span>
        <span style="color: #38BDF8;">Cosine Similarity (768d)</span>
      </div>
      <div style="margin-top: 14px;">
        {gemma_bars_html}
      </div>
    </div>

    <!-- mini-Jev Options Chart -->
    <div class="card">
      <div class="card-title">
        <span>mini-Jev Decision Controller Softmax Option Scores</span>
        <span style="color: #A855F7;">K-way Choice Head</span>
      </div>
      <div style="margin-top: 14px;">
        {jev_bars_html}
      </div>
    </div>
  </div>

  <!-- Live Audit Trace Table -->
  <div class="card">
    <div class="card-title">
      <span>Live Audit Trace Feed (.alienese/opencode_trace.jsonl)</span>
      <span style="color: #10B981;">{data["trace_total_count"]} Total Events</span>
    </div>
    <div style="overflow-x: auto; margin-top: 10px;">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Phase</th>
            <th>Tool</th>
            <th>Arguments (Redacted)</th>
            <th>State Digest</th>
          </tr>
        </thead>
        <tbody id="trace-rows">
          {trace_rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    // Live stream via Server-Sent Events (SSE) respecting session_id
    const urlParams = new URLSearchParams(window.location.search);
    const sid = urlParams.get('session_id') || "{data['session_id']}";
    const streamUrl = sid ? '/api/stream?session_id=' + encodeURIComponent(sid) : '/api/stream';
    const eventSource = new EventSource(streamUrl);
    eventSource.onmessage = function(event) {{
      try {{
        const payload = JSON.parse(event.data);
        if (payload && payload.timestamp) {{
          if (window.__last_step !== undefined && window.__last_step !== payload.step_count) {{
            window.location.reload();
          }}
          window.__last_step = payload.step_count;
        }}
      }} catch(e) {{}}
    }};
    eventSource.onerror = function() {{
      setTimeout(() => window.location.reload(), 3000);
    }};
  </script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving HTML, JSON status, and SSE updates."""

    workspace: Path = DEFAULT_WORKSPACE

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard access logs to avoid terminal clutter."""
        pass

    def _get_target_session_id(self) -> str | None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        sids = params.get("session_id")
        return sids[0] if sids else None

    def do_HEAD(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        session_id = self._get_target_session_id()
        if parsed.path in ("/", "/index.html"):
            data = read_workspace_data(self.workspace, target_session_id=session_id)
            html_content = generate_html_dashboard(data).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content)))
            self.end_headers()
        elif parsed.path in ("/api/status", "/api/metrics", "/api/events", "/api/plan"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
        else:
            self.send_response(HTTPStatus.OK)
            self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        session_id = self._get_target_session_id()

        if parsed.path in ("/", "/index.html"):
            data = read_workspace_data(self.workspace, target_session_id=session_id)
            html_content = generate_html_dashboard(data).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content)))
            self.end_headers()
            self.wfile.write(html_content)

        elif parsed.path == "/api/status":
            data = read_workspace_data(self.workspace, target_session_id=session_id)
            json_bytes = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(json_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json_bytes)

        elif parsed.path == "/api/metrics":
            data = read_workspace_data(self.workspace, target_session_id=session_id)
            metrics_payload = {
                "session_id": data.get("session_id"),
                "candidate_context_reduction": data.get("candidate_context_reduction"),
                "actual_api_savings": data.get("actual_api_savings"),
                "tokens_per_model": data.get("tokens_per_model"),
                "models": data.get("models"),
            }
            json_bytes = json.dumps(metrics_payload, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(json_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json_bytes)

        elif parsed.path == "/api/events":
            data = read_workspace_data(self.workspace, target_session_id=session_id)
            events_payload = {
                "session_id": data.get("session_id"),
                "events": data.get("trace_events", []),
                "total_count": data.get("trace_total_count", 0),
            }
            json_bytes = json.dumps(events_payload, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(json_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json_bytes)

        elif parsed.path == "/api/plan":
            data = read_workspace_data(self.workspace, target_session_id=session_id)
            plan_payload = {
                "session_id": data.get("session_id"),
                "ranked_files": data.get("ranked_files", []),
                "jev_options": data.get("jev_options", []),
                "selected_id": data.get("selected_id"),
                "selected_label": data.get("selected_label"),
                "selected_probability": data.get("selected_probability"),
                "adherence": data.get("adherence"),
            }
            json_bytes = json.dumps(plan_payload, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(json_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json_bytes)

        elif parsed.path == "/api/stream":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for _ in range(3600):  # Stream up to 1 hour
                    data = read_workspace_data(self.workspace, target_session_id=session_id)
                    curr_step = data.get("step_count", 0)
                    msg_payload = json.dumps(
                        {
                            "timestamp": time.time(),
                            "session_id": data.get("session_id"),
                            "step_count": curr_step,
                            "verification_status": data.get("verification_status"),
                            "mutations_count": len(data.get("mutations", [])),
                        }
                    )
                    self.wfile.write(f"data: {msg_payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()


def run_dashboard_server(
    port: int = DEFAULT_PORT,
    workspace: Path = DEFAULT_WORKSPACE,
) -> None:
    """Run dashboard HTTP/SSE companion server."""
    DashboardRequestHandler.workspace = workspace
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardRequestHandler)
    print(f"[Alienese Visual Dashboard] Serving at http://127.0.0.1:{port}/")
    print(f"[Alienese Visual Dashboard] Monitoring workspace: {workspace}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Alienese Visual Dashboard] Shutting down...")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Alienese Visual Companion Dashboard")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 4097)"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Target workspace to monitor (default: /tmp/alienese_opencode_demo)",
    )
    args = parser.parse_args()
    run_dashboard_server(port=args.port, workspace=args.workspace)


if __name__ == "__main__":
    main()
