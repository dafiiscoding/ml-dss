import os
import numpy as np
from PIL import Image
from src.config import CLIP_MODEL_NAME

try:
    import torch
    HAS_TORCH = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    _DEVICE = "cpu"

# Global holders to avoid reloading multiple times
_CLIP_PROCESSOR = None
_CLIP_MODEL = None

def load_clip_model():
    """
    Lazy load CLIP model. Falls back to None if model download fails or torch is missing.
    """
    global _CLIP_PROCESSOR, _CLIP_MODEL
    if not HAS_TORCH:
        return None, None

    if _CLIP_PROCESSOR is None or _CLIP_MODEL is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            print(f"Loading CLIP model '{CLIP_MODEL_NAME}' on {_DEVICE}...")
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            _CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_DEVICE)
            _CLIP_MODEL.eval()
            print("CLIP model loaded successfully.")
        except Exception as e:
            print(f"Failed to load CLIP model due to: {e}. Fallback options will be used.")
            _CLIP_PROCESSOR = None
            _CLIP_MODEL = None
    return _CLIP_PROCESSOR, _CLIP_MODEL

def _clip_image_embeds(model, inputs):
    """
    Return the projected 512-d CLIP image embedding, robust across transformers
    versions. Old API: get_image_features returns the projected tensor directly.
    transformers 5.x: it returns a vision output object whose pooler_output
    (768-d) must be passed through model.visual_projection to reach CLIP space.
    """
    out = model.get_image_features(**inputs)
    import torch as _t
    if isinstance(out, _t.Tensor):
        return out
    # transformers 5.x: the returned object's pooler_output is already the
    # projected 512-d CLIP image embedding.
    if getattr(out, "image_embeds", None) is not None:
        return out.image_embeds
    return out.pooler_output


def _clip_text_embeds(model, inputs):
    """Return the projected 512-d CLIP text embedding."""
    out = model.get_text_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out
    if getattr(out, "text_embeds", None) is not None:
        return out.text_embeds
    return out.pooler_output


def extract_text_embeddings(texts, batch_size=128):
    """Extract normalized CLIP embeddings for cleaned tweet text."""
    processor, model = load_clip_model()
    if not (HAS_TORCH and processor is not None and model is not None):
        raise RuntimeError(
            "CLIP is unavailable. Install torch/transformers and ensure "
            f"'{CLIP_MODEL_NAME}' can be loaded before extracting embeddings."
        )

    embeddings = []
    print(f"Extracting CLIP text features for {len(texts)} rows...")
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        try:
            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(_DEVICE)
            with torch.no_grad():
                features = _clip_text_embeds(model, inputs)
                features = features / features.norm(
                    p=2, dim=-1, keepdim=True
                ).clamp_min(1e-12)
                embeddings.append(features.cpu().numpy())
        except Exception as exc:
            raise RuntimeError(
                f"CLIP text forward pass failed for batch starting at {start}"
            ) from exc
    return np.vstack(embeddings)


def extract_image_embeddings(image_paths, base_dir, batch_size=16):
    """
    Extract CLIP embeddings for a list of image paths.

    The academic pipeline fails loudly when CLIP is unavailable. Silent
    color/dummy fallbacks would make generated metrics look real when they are
    not comparable to the declared method.
    """
    processor, model = load_clip_model()

    embeddings = []

    if HAS_TORCH and processor is not None and model is not None:
        # CLIP Extraction
        print(f"Extracting features for {len(image_paths)} images using CLIP...")
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            for path in batch_paths:
                abs_path = os.path.join(base_dir, path)
                try:
                    img = Image.open(abs_path).convert("RGB")
                    # Ensure resize
                    img = img.resize((224, 224))
                    batch_images.append(img)
                except Exception as e:
                    raise RuntimeError(f"Cannot load referenced image: {abs_path}") from e

            try:
                inputs = processor(images=batch_images, return_tensors="pt").to(_DEVICE)
                with torch.no_grad():
                    features = _clip_image_embeds(model, inputs)
                    # L2-normalize embeddings
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                    embeddings.append(features.cpu().numpy())
            except Exception as e:
                raise RuntimeError(
                    f"CLIP forward pass failed for batch starting at {i}"
                ) from e

        return np.vstack(embeddings)
    raise RuntimeError(
        "CLIP is unavailable. Install torch/transformers and ensure "
        f"'{CLIP_MODEL_NAME}' can be loaded before extracting embeddings."
    )

if __name__ == "__main__":
    # Test file
    dummy_img_path = os.path.join("data_image", "test_sample.png")
    os.makedirs(os.path.dirname(dummy_img_path), exist_ok=True)
    img = Image.new('RGB', (224, 224), color=(255, 0, 0))
    img.save(dummy_img_path)

    # Extract
    embeds = extract_image_embeddings([dummy_img_path], os.path.abspath("."))
    print(f"Features shape: {embeds.shape}")
    print(f"First 5 values: {embeds[0, :5]}")
