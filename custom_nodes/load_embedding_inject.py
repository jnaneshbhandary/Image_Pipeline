"""
File name: custom_nodes/load_embedding_inject.py

Purpose:
ComfyUI custom node that loads a saved 512-dimensional ArcFace identity
embedding and injects it into an IPAdapter FaceID model object, bypassing
CLIP Vision during image generation.

How to run it:
Run `python setup.py` to copy this file into `ComfyUI/custom_nodes/`, then
start ComfyUI normally. ComfyUI loads the node automatically at startup.

Prerequisites:
Run `python remember_face.py` first to create `identity/character.npy`, and
install ComfyUI_IPAdapter_plus pinned to the expected commit via setup.py.

Expected runtime:
Loaded at ComfyUI startup. Each node execution should take less than a second.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent
_SENTINEL = object()


def resolve_embedding_path(embedding_path: str) -> Path:
    path = Path(embedding_path)
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    comfy_parent_candidate = (Path.cwd().parent / path).resolve()
    if Path.cwd().name.lower() == "comfyui" and comfy_parent_candidate.exists():
        return comfy_parent_candidate

    project_parent_candidate = (PROJECT_ROOT.parent / path).resolve()
    if project_parent_candidate.exists():
        return project_parent_candidate

    return cwd_candidate


class LoadEmbeddingAndInject:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "embedding_path": ("STRING", {"default": "identity/character.npy"}),
                "ipadapter_model": ("IPADAPTER",),
                "weight": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "weight_type": (["linear", "ease in", "ease out", "ease in-out"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "attn_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IPADAPTER",)
    RETURN_NAMES = ("ipadapter_with_embedding",)
    FUNCTION = "inject"
    CATEGORY = "ipadapter/face"
    DISPLAY_NAME = "Load Embedding and Inject (FaceID)"

    def inject(
        self,
        embedding_path: str,
        ipadapter_model: Any,
        weight: float,
        weight_type: str,
        start_at: float,
        end_at: float,
        attn_mask: Any | None = None,
    ) -> tuple[Any]:
        resolved_path = resolve_embedding_path(embedding_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Embedding file not found at {resolved_path}. Run remember_face.py first."
            )

        raw = np.load(str(resolved_path))
        if raw.shape != (512,):
            raise ValueError(f"Expected embedding shape (512,), got {raw.shape}. Re-run remember_face.py.")

        embedding = torch.from_numpy(raw).float()
        norm_before = embedding.norm().item()
        if norm_before == 0.0:
            raise ValueError("Embedding norm is zero. The .npy file may be corrupt. Re-run remember_face.py.")
        embedding = embedding / embedding.norm()
        norm_after = embedding.norm().item()
        logger.info(
            "[LoadEmbeddingAndInject] Loaded embedding. Norm before re-normalise: "
            "%.6f. Norm after: %.6f (expected 1.000000)",
            norm_before,
            norm_after,
        )
        if abs(norm_after - 1.0) >= 0.001:
            raise ValueError(
                f"Embedding normalisation failed. Norm is {norm_after:.6f}. "
                "The .npy file may be corrupt. Re-run remember_face.py."
            )

        embedding = embedding.unsqueeze(0).unsqueeze(0)
        try:
            device = next(iter(ipadapter_model.parameters())).device
            embedding = embedding.to(device)
        except StopIteration:
            logger.debug("IPAdapter model parameters() iterator is empty; keeping embedding on CPU.")
        except AttributeError:
            logger.warning("IPAdapter model has no parameters() method; keeping embedding on CPU.")
        except TypeError as exc:
            logger.warning("Could not inspect IPAdapter model parameters: %s", exc)

        model_copy = copy.deepcopy(ipadapter_model)

        inject_attrs: dict[str, Any] = {
            "_precomputed_embeds": embedding,
            "_weight": weight,
            "_weight_type": weight_type,
            "_start_at": start_at,
            "_end_at": end_at,
        }
        if attn_mask is not None:
            inject_attrs["_attn_mask"] = attn_mask

        for attr_name, value in inject_attrs.items():
            setattr(model_copy, attr_name, value)
            retrieved = getattr(model_copy, attr_name, _SENTINEL)
            if retrieved is _SENTINEL:
                raise RuntimeError(
                    f"[LoadEmbeddingAndInject] INJECTION FAILED: attribute "
                    f"'{attr_name}' could not be set on the IPAdapter model. "
                    f"This means the internal API of ComfyUI_IPAdapter_plus has "
                    f"changed. Check which version is installed at: "
                    f"ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/ and update "
                    f"the attribute names in load_embedding_inject.py to match. "
                    f"Expected commit: 4e1a0fd"
                )

        logger.info(
            "[LoadEmbeddingAndInject] Injection verified. "
            "Embedding norm: %.6f "
            "(must be ~1.000000 for identity to lock). "
            "Weight: %s. Weight type: %s. "
            "Active range: %s-%s.",
            embedding.norm().item(),
            weight,
            weight_type,
            start_at,
            end_at,
        )
        return (model_copy,)


NODE_CLASS_MAPPINGS = {
    "LoadEmbeddingAndInject": LoadEmbeddingAndInject,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadEmbeddingAndInject": "Load Embedding and Inject (FaceID)",
}
