"""Sora Nano V3 prompt contract: explicit structured design requests.

This is a conservative *language interface*, not a trained image model.
Unknown attributes are rejected instead of silently generating a wrong image.
The resulting specification is intended for future learned conditioning.
"""
from dataclasses import asdict, dataclass
import json
import re


class UnsupportedDesignPrompt(ValueError):
    pass


@dataclass(frozen=True)
class DesignSpec:
    shape: str
    color: str
    corner_radius: float
    language: str
    object_count: int = 1
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)


COLOR_WORDS = {
    "red": ("red", "أحمر", "حمراء", "احمر", "حمراء"),
    "blue": ("blue", "أزرق", "زرقاء", "ازرق"),
    "green": ("green", "أخضر", "خضراء", "اخضر"),
    "yellow": ("yellow", "أصفر", "صفراء", "اصفر"),
}
SHAPE_WORDS = {
    "square": ("square", "مربع"),
    "circle": ("circle", "دائرة", "دائره"),
    "triangle": ("triangle", "مثلث"),
}
CORNER_MODIFIERS = {
    0.25: ("slightly rounded", "small rounded", "slightly rounded corners", "small rounded corners", "rounded a little",
           "زوايا دائرية صغيرة", "زوايا مستديرة صغيرة", "حواف دائرية صغيرة"),
    0.5: ("rounded corners", "rounded square", "زوايا دائرية", "زوايا مستديرة",
          "حواف مستديرة", "حواف دائرية"),
    0.75: ("very rounded", "large rounded", "heavily rounded",
           "large radius", "large rounded corners", "very rounded corners", "square with large rounded corners", "زوايا دائرية كبيرة", "زوايا مستديرة كبيرة",
           "حواف مستديرة كبيرة", "حواف دائرية كبيرة"),
}
FILLER = ("create", "make", "draw", "generate", "design", "please",
          "a", "an", "the", "with", "and", "for", "me", "of",
          "انشئ", "أنشئ", "ارسم", "صمم", "اصنع", "لي", "من", "مع",
          "لون", "بلون", "ذو", "ذات", "شكل", "بزوايا", "بـ")


def _words(text):
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def _find_phrases(text, phrases):
    found = []
    for label, variants in phrases.items():
        for variant in variants:
            regex = r"(?<!\w)" + r"\s+".join(map(re.escape, variant.split())) + r"(?!\w)"
            match = re.search(regex, text)
            if match:
                found.append((label, match.span(), variant))
    return found


def parse_design_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise UnsupportedDesignPrompt("Expected non-empty prompt")
    text = re.sub(r"\s+", " ", prompt.strip().lower())
    language = "ar" if re.search(r"[\u0600-\u06ff]", text) else "en"
    shapes = _find_phrases(text, SHAPE_WORDS)
    colors = _find_phrases(text, COLOR_WORDS)
    if len(set(x[0] for x in shapes)) != 1 or len(set(x[0] for x in colors)) != 1:
        raise UnsupportedDesignPrompt("Exactly one supported shape and color required")
    shape, color = shapes[0][0], colors[0][0]
    corners = _find_phrases(text, CORNER_MODIFIERS)
    # Prefer the longest modifier, and reject conflicting equal-length modifiers.
    corners.sort(key=lambda x: len(x[2]), reverse=True)
    radius = corners[0][0] if corners else 0.0
    if shape != "square" and corners:
        raise UnsupportedDesignPrompt("Rounded corner modifier only trained for square")
    # Reject unexplained content, including textures, position, plurality,
    # and unsupported modifiers. This prevents false claims of understanding.
    consumed = [False] * len(text)
    for _, (start, end), _ in shapes + colors + corners:
        for i in range(start, end):
            consumed[i] = True
    remainder = "".join(" " if used else text[i] for i, used in enumerate(consumed))
    allowed = set(FILLER)
    unknown = [word for word in _words(remainder) if word not in allowed]
    if unknown:
        raise UnsupportedDesignPrompt("Unsupported prompt content: " + ", ".join(unknown))
    return DesignSpec(shape=shape, color=color, corner_radius=radius,
                      language=language)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps({"status": "parsed_not_generated",
                          "spec": parse_design_prompt(args.prompt).to_dict(),
                          "image_generation_supported": False},
                         ensure_ascii=False, indent=2))
    except UnsupportedDesignPrompt as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
