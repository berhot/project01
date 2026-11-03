"""Sora Nano synthetic dataset v2: 12 combinations, two languages.

TRAINING and synthetic data previews only: no pretrained assets, downloads,
external model, or storage of millions of images. Draws on demand in memory.
The RGB diffusion model generates its own pixels at inference.
"""
import random

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.pixel_diffusion import WORDS as V1_WORDS

DATASET_VERSION = "v2"
COLORS = {
    "red": (230, 45, 50),
    "green": (36, 190, 85),
    "blue": (45, 90, 225),
    "yellow": (239, 201, 39),
}
SHAPES = ("circle", "square", "triangle")
LIGHT_BACKGROUNDS = (
    (246, 246, 244),  # warm white
    (237, 240, 243),  # light gray
    (250, 241, 230),  # warm cream
)
AR_SHAPES = {"circle": "دائرة", "square": "مربع", "triangle": "مثلث"}
AR_COLORS = {
    "circle": {
        "red": "حمراء", "green": "خضراء", "blue": "زرقاء", "yellow": "صفراء"
    },
    "square": {
        "red": "أحمر", "green": "أخضر", "blue": "أزرق", "yellow": "أصفر"
    },
    "triangle": {
        "red": "أحمر", "green": "أخضر", "blue": "أزرق", "yellow": "أصفر"
    },
}
# Append only; retain every exact v1 embedding index.
EXTRA_WORDS = (
    "yellow", "triangle", "light", "مثلث", "أصفر", "صفراء", "فاتحة",
)
WORDS_V2 = V1_WORDS + EXTRA_WORDS
assert len(set(WORDS_V2)) == len(WORDS_V2)
CONCEPTS = tuple((color, shape) for shape in SHAPES for color in COLORS)


def caption_v2(color, shape, language="en", variant=0):
    if color not in COLORS or shape not in SHAPES:
        raise ValueError("Unsupported v2 shape/color")
    if language == "en":
        captions = (
            f"{color} {shape}",
            f"a {color} {shape}",
            f"simple {color} {shape} on light background",
        )
    elif language == "ar":
        noun = AR_SHAPES[shape]
        adjective = AR_COLORS[shape][color]
        captions = (
            f"{noun} {adjective}",
            f"شكل {noun} {adjective} بسيط",
            f"{noun} {adjective} على خلفية فاتحة",
        )
    else:
        raise ValueError("Language must be en or ar")
    return captions[variant % len(captions)]


def tokenize_v2(prompt, max_tokens=12):
    from src.pixel_diffusion import tokenize
    return tokenize(prompt, max_tokens=max_tokens, dataset_version="v2")


def synthetic_example_v2(seed, color=None, shape=None, language=None,
                         split="train", variant=None):
    """Deterministic seed-indexed bilingual RGB samples; no disk usage.

    Validation examples have a separate PRNG seed space. All 12 semantic
    classes exist in both splits, and visual placements are independently
    sampled. This is not a real-world generalization benchmark.
    """
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    if color is not None and color not in COLORS:
        raise ValueError("Invalid color")
    if shape is not None and shape not in SHAPES:
        raise ValueError("Invalid shape")
    if language is not None and language not in ("ar", "en"):
        raise ValueError("Invalid language")
    shift = 0 if split == "train" else 1_000_000_007
    rng = random.Random(int(seed) + shift)
    color = color if color is not None else rng.choice(tuple(COLORS))
    shape = shape if shape is not None else rng.choice(SHAPES)
    language = language if language is not None else rng.choice(("ar", "en"))
    variant = rng.randrange(3) if variant is None else int(variant)
    prompt = caption_v2(color, shape, language, variant)

    image = Image.new("RGB", (32, 32), rng.choice(LIGHT_BACKGROUNDS))
    pen = ImageDraw.Draw(image)
    side = rng.randint(11, 23)
    x = rng.randint(2, 30 - side)
    y = rng.randint(2, 30 - side)
    bbox = (x, y, x + side, y + side)
    if shape == "circle":
        pen.ellipse(bbox, fill=COLORS[color])
    elif shape == "square":
        pen.rectangle(bbox, fill=COLORS[color])
    else:
        # Two orientations, same caption; image is still one triangle.
        if rng.randrange(2):
            vertices = [(x, y), (x + side, y), (x + side / 2, y + side)]
        else:
            vertices = [(x + side / 2, y), (x + side, y + side),
                        (x, y + side)]
        pen.polygon(vertices, fill=COLORS[color])

    pixels = np.asarray(image, dtype=np.float32).copy()
    image_tensor = torch.from_numpy(pixels).permute(2, 0, 1) / 127.5 - 1.0
    return image_tensor, tokenize_v2(prompt), prompt, (color, shape)


def balanced_training_batch(step, batch_size, seed=42):
    """Round-robin 12 classes; over multiples of 12 exactly balanced."""
    if step < 1 or batch_size < 1:
        raise ValueError("Invalid step or batch size")
    # Use changing class offsets for small (non-multiple of 12) batches too.
    offset = (step - 1) * batch_size
    images, tokens = [], []
    for index in range(batch_size):
        color, shape = CONCEPTS[(offset + index) % len(CONCEPTS)]
        language = "ar" if ((step + index) % 2) else "en"
        image, token, _, _ = synthetic_example_v2(
            seed + step * 4099 + index * 13, color=color, shape=shape,
            language=language, split="train",
        )
        images.append(image)
        tokens.append(token)
    return torch.stack(images), torch.stack(tokens)
