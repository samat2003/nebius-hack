"""Local Model Server hosting EmbeddingGemma 2 and mini-Jev for Alienese.

Serves:
- `POST /embeddings` for EmbeddingGemma 2 (`google/embeddinggemma-2`)
- `POST /predict` for mini-Jev (`samatv256/mini-Jev` @ revision `step-010626`)
- `GET /health` for readiness and device telemetry

Enforces genuine weights and model inference, running in `.venv-models`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("alienese.model_server")

app = FastAPI(title="Alienese Local Model Server", version="0.1.0")

# Global model state
_retriever_model: Any = None
_mini_jev_instance: Any = None
_device: str = "cpu"


class EmbeddingRequest(BaseModel):
    model: str = "google/embeddinggemma-2"
    input: list[str]
    dimensions: int | None = 768


class PredictRequest(BaseModel):
    model: str = "samatv256/mini-Jev"
    revision: str = "step-010626"
    state: dict[str, Any] | str
    question: str
    question_type: str = "choice"
    answer_options: list[dict[str, Any]]


class LocalMiniJev:
    """CPU/CUDA compatible loader for samatv256/mini-Jev."""

    def __init__(self, release_dir: Path, device: str = "cpu") -> None:
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        # Add release_dir to sys.path to import inference helper classes
        sys.path.insert(0, str(release_dir))
        from inference import _SetHead, encode_branch  # type: ignore

        self._encode_branch = encode_branch
        self.device = device

        config_path = release_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_id = config["base_model"]
        revision = config["base_revision"]

        logger.info(f"Loading Qwen3-0.6B tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        logger.info(f"Loading base backbone {model_id} on {device}...")
        base = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        )
        if device != "cpu":
            base = base.to(device)

        logger.info(f"Attaching LoRA adapter from {release_dir / config['adapter_dir']}...")
        self.encoder = PeftModel.from_pretrained(
            base,
            release_dir / config["adapter_dir"],
            is_trainable=False,
        ).eval()
        self.backbone = self.encoder.base_model.model

        width = int(config["projection_dim"])
        self.projection = torch.nn.Linear(self.backbone.config.hidden_size, width).to(device)
        self.scorer = _SetHead(
            width,
            int(config["set_layers"]),
            int(config["set_heads"]),
        ).to(device)

        logger.info(f"Loading decision head from {release_dir / config['head_file']}...")
        weights = load_file(release_dir / config["head_file"], device="cpu")
        self.projection.load_state_dict(
            {
                k.removeprefix("projection."): v.to(device)
                for k, v in weights.items()
                if k.startswith("projection.")
            },
            strict=True,
        )
        self.scorer.load_state_dict(
            {
                k.removeprefix("scorer."): v.to(device)
                for k, v in weights.items()
                if k.startswith("scorer.")
            },
            strict=True,
        )

        self.projection.eval()
        self.scorer.eval()
        self.max_tokens = int(config["max_tokens"])
        logger.info("mini-Jev loaded and ready for decision inference.")

    def predict(
        self,
        state: dict[str, Any] | str,
        question: str,
        question_type: str,
        answer_options: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(answer_options) < 2:
            raise ValueError("provide at least two answer options")
        labels = [str(opt.get("label") or opt.get("name") or "") for opt in answer_options]
        if any(not label for label in labels):
            raise ValueError("every option needs a label")
        option_ids = [str(opt.get("id", i)) for i, opt in enumerate(answer_options)]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("option IDs must be unique")

        vectors = []
        lengths = []
        with torch.inference_mode():
            for opt in answer_options:
                ids, suffix_mask = self._encode_branch(
                    self.tokenizer,
                    state,
                    question,
                    question_type,
                    opt,
                    self.max_tokens,
                )
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                mask = torch.tensor([suffix_mask], dtype=torch.bool, device=self.device)
                hidden = self.backbone(input_ids=input_ids, use_cache=False).last_hidden_state
                pooled = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                vectors.append(self.projection(pooled).squeeze(0))
                lengths.append(len(ids))

            valid = torch.ones((1, len(vectors)), dtype=torch.bool, device=self.device)
            logits = self.scorer(torch.stack(vectors).unsqueeze(0), valid)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()

        winner = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "selected_id": option_ids[winner],
            "selected_label": labels[winner],
            "options": [
                {"id": option_ids[i], "label": labels[i], "probability": probs[i]}
                for i in range(len(probs))
            ],
            "branch_tokens": lengths,
        }


def init_models(mini_jev_dir: str | None = None) -> None:
    global _retriever_model, _mini_jev_instance, _device

    # Use CPU by default to ensure maximum stability across WSL CUDA configurations
    _device = "cpu"
    logger.info(f"Targeting device: {_device}")

    # 1. EmbeddingGemma 2
    from sentence_transformers import SentenceTransformer

    logger.info("Initializing SentenceTransformer('google/embeddinggemma-2')...")
    _retriever_model = SentenceTransformer(
        "google/embeddinggemma-2",
        device=_device,
    )
    logger.info("EmbeddingGemma 2 ready.")

    # 2. mini-Jev
    if mini_jev_dir:
        snap_path = Path(mini_jev_dir)
    else:
        logger.info("Downloading samatv256/mini-Jev snapshot (step-010626)...")
        snap = snapshot_download(
            repo_id="samatv256/mini-Jev",
            revision="step-010626",
        )
        snap_path = Path(snap)

    _mini_jev_instance = LocalMiniJev(snap_path, device=_device)
    logger.info("All local models initialized successfully.")


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "retriever": {
            "model": "google/embeddinggemma-2",
            "status": "ready" if _retriever_model is not None else "uninitialized",
            "device": _device,
        },
        "controller": {
            "model": "samatv256/mini-Jev",
            "revision": "step-010626",
            "status": "ready" if _mini_jev_instance is not None else "uninitialized",
            "device": _device,
        },
    }


@app.post("/embeddings")
async def create_embeddings(req: EmbeddingRequest) -> dict[str, Any]:
    if _retriever_model is None:
        raise HTTPException(status_code=503, detail="EmbeddingGemma retriever not initialized.")

    if not req.input:
        raise HTTPException(status_code=400, detail="Input list must not be empty.")

    try:
        raw_embeddings = _retriever_model.encode(
            req.input,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
    except Exception as exc:
        logger.error(f"Failed to generate embeddings: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    target_dim = req.dimensions or 768
    data = []
    for idx, vec in enumerate(raw_embeddings):
        truncated = vec[:target_dim].tolist()
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in truncated))
        normed_vec = [v / norm for v in truncated] if norm > 1e-12 else truncated
        data.append({"index": idx, "embedding": normed_vec})

    return {
        "model": req.model,
        "model_revision": "gemma2-emb-v1",
        "data": data,
        "usage": {
            "prompt_tokens": sum(len(s.split()) for s in req.input),
            "total_tokens": sum(len(s.split()) for s in req.input),
        },
    }


@app.post("/predict")
async def predict_decision(req: PredictRequest) -> dict[str, Any]:
    if _mini_jev_instance is None:
        raise HTTPException(status_code=503, detail="mini-Jev controller not initialized.")

    if len(req.answer_options) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two answer options.")

    try:
        pred = _mini_jev_instance.predict(
            state=req.state,
            question=req.question,
            question_type=req.question_type,
            answer_options=req.answer_options,
        )
    except Exception as exc:
        logger.error(f"Decision prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "model": "samatv256/mini-Jev",
        "model_revision": "step-010626",
        "selected_id": pred["selected_id"],
        "selected_label": pred.get("selected_label"),
        "options": pred["options"],
        "branch_tokens": pred.get("branch_tokens", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alienese Local Model Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001)")
    parser.add_argument(
        "--mini-jev-dir",
        default="/home/samat_zharassov/.cache/huggingface/hub/models--samatv256--mini-Jev/snapshots/3a1d1d19d85e9863146307fe4b769e8bbe242c4e",
        help="Path to mini-Jev release snapshot directory",
    )
    args = parser.parse_args()

    init_models(mini_jev_dir=args.mini_jev_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
