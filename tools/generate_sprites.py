"""
Sprite Generator for Aria — Chibi Kurisu Makise
================================================
Generates 6 expression sprites at 256x256 RGBA using Pillow.
Overwrites existing files in assets/sprites/.

Character: Chibi anime girl with auburn hair, red headphones,
round glasses, lab coat — Kurisu Makise (Steins;Gate) inspired.

Usage:
    python tools/generate_sprites.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image, ImageDraw, ImageFont

SPRITE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "sprites")
SIZE = 256

# ── Colour Palette ──────────────────────────────────────────────────
HAIR        = (195, 90, 31, 255)     # #C35A1F
HAIR_SHADOW = (139, 62, 18, 255)     # #8B3E12
SKIN        = (253, 219, 180, 255)   # #FDDBB4
SKIN_DARK   = (240, 192, 144, 60)    # #F0C090 at ~60 alpha
EYE_IRIS    = (200, 112, 134, 255)   # #C87086
EYE_PUPIL   = (60, 21, 24, 255)      # #3C1518
EYE_WHITE   = (255, 255, 255, 255)
GLASSES     = (122, 92, 56, 255)     # #7A5C38
BLUSH       = (255, 176, 176, 80)    # #FFB0B0 at alpha 80
BLUSH_STRONG = (255, 176, 176, 120)  # Stronger blush for sleep
BROW        = (107, 58, 31, 255)     # #6B3A1F
MOUTH_LINE  = (192, 112, 112, 255)   # #C07070
MOUTH_OPEN  = (60, 21, 24, 255)      # #3C1518
HPHONE_RED  = (204, 34, 34, 255)     # #CC2222
HPHONE_DARK = (30, 30, 30, 255)      # #1E1E1E
HPHONE_INNER = (68, 17, 17, 255)     # #441111
COAT_WHITE  = (245, 245, 245, 255)   # #F5F5F5
COAT_SHADOW = (220, 220, 220, 255)   # #DCDCDC
TOP_DARK    = (44, 44, 44, 255)      # #2C2C2C
OUTLINE     = (42, 26, 10, 255)      # #2A1A0A
TRANSPARENT = (0, 0, 0, 0)
ZZZ_COLOUR  = (160, 160, 255, 200)   # #A0A0FF


def _draw_hair_back(draw: ImageDraw.Draw) -> None:
    """Layer 1: Long hair hanging behind the head on both sides."""
    # Left hair strand
    draw.ellipse([52, 75, 118, 248], fill=HAIR, outline=HAIR_SHADOW, width=2)
    # Right hair strand
    draw.ellipse([138, 75, 204, 248], fill=HAIR, outline=HAIR_SHADOW, width=2)
    # Connecting rectangle between the two sides
    draw.rectangle([90, 180, 166, 248], fill=HAIR)


def _draw_head(img: Image.Image, draw: ImageDraw.Draw) -> None:
    """Layer 2: Head/face ellipse with subtle chin shading."""
    # Main face
    draw.ellipse([128 - 66, 108 - 60, 128 + 66, 108 + 60],
                 fill=SKIN, outline=OUTLINE, width=2)
    # Chin shading (semi-transparent overlay)
    overlay = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.ellipse([128 - 30, 148 - 16, 128 + 30, 148 + 16], fill=SKIN_DARK)
    img.alpha_composite(overlay)


def _draw_bangs(draw: ImageDraw.Draw) -> None:
    """Layer 3: Fringe/bangs across the forehead.

    Uses overlapping ellipses to create a softer, more natural look
    instead of the hard-edged rectangle approach.
    """
    # Main bangs body — wide ellipse covering the forehead
    draw.ellipse([60, 42, 196, 108], fill=HAIR)
    # Flatten the top of the bangs with a slightly narrower ellipse
    draw.ellipse([65, 38, 191, 95], fill=HAIR)
    # Subtle bottom edge shadow for depth
    draw.arc([63, 78, 193, 118], start=10, end=170, fill=HAIR_SHADOW, width=1)


def _draw_ahoge(draw: ImageDraw.Draw) -> None:
    """Layer 4: Iconic hair strand sticking up from the crown."""
    draw.polygon(
        [(124, 60), (134, 60), (152, 20), (144, 14)],
        fill=HAIR, outline=HAIR_SHADOW,
    )


def _draw_headphone_band(draw: ImageDraw.Draw) -> None:
    """Layer 5: Dark arc headphone band over the top of the head."""
    draw.arc([72, 50, 184, 130], start=200, end=340,
             fill=HPHONE_DARK, width=7)


def _draw_ears(draw: ImageDraw.Draw) -> None:
    """Layer 6: Small skin-coloured ear ovals on each side."""
    draw.ellipse([60, 98, 76, 122], fill=SKIN, outline=OUTLINE, width=1)
    draw.ellipse([180, 98, 196, 122], fill=SKIN, outline=OUTLINE, width=1)


def _draw_headphone_cups(draw: ImageDraw.Draw) -> None:
    """Layer 7: Prominent red headphone cups over each ear."""
    for cx in (62, 194):
        # Outer red cup
        draw.ellipse([cx - 26, 110 - 26, cx + 26, 110 + 26],
                     fill=HPHONE_RED, outline=OUTLINE, width=2)
        # Inner ring
        draw.ellipse([cx - 18, 110 - 18, cx + 18, 110 + 18],
                     fill=HPHONE_INNER)
        # Centre dark circle
        draw.ellipse([cx - 10, 110 - 10, cx + 10, 110 + 10],
                     fill=HPHONE_DARK)


def _draw_eyes_open(draw: ImageDraw.Draw) -> None:
    """Layer 8 (open): Standard open eyes with iris, pupil, highlight."""
    for side, cx in [("left", 107), ("right", 149)]:
        # Iris
        draw.ellipse([cx - 16, 97, cx + 16, 123],
                     fill=EYE_IRIS, outline=OUTLINE, width=1)
        # Pupil (slightly below centre)
        draw.ellipse([cx - 6, 112 - 6, cx + 6, 112 + 6], fill=EYE_PUPIL)
        # Highlight (top-left of pupil)
        hx = cx - 5
        draw.ellipse([hx - 3, 107 - 3, hx + 3, 107 + 3], fill=EYE_WHITE)
        # Lower lash line
        draw.arc([cx - 16, 110, cx + 16, 128], start=200, end=340,
                 fill=OUTLINE, width=1)


def _draw_eyes_wide(draw: ImageDraw.Draw) -> None:
    """Layer 8 (wide): Slightly taller eyes for the wake/surprised state."""
    for side, cx in [("left", 107), ("right", 149)]:
        # Iris — 10% taller (+2px top and bottom)
        draw.ellipse([cx - 16, 95, cx + 16, 125],
                     fill=EYE_IRIS, outline=OUTLINE, width=1)
        # Pupil
        draw.ellipse([cx - 6, 112 - 6, cx + 6, 112 + 6], fill=EYE_PUPIL)
        # Highlight
        hx = cx - 5
        draw.ellipse([hx - 3, 107 - 3, hx + 3, 107 + 3], fill=EYE_WHITE)
        # Lower lash line
        draw.arc([cx - 16, 110, cx + 16, 130], start=200, end=340,
                 fill=OUTLINE, width=1)


def _draw_eyes_closed(draw: ImageDraw.Draw) -> None:
    """Layer 8 (closed): Curved closed-eye lines for blink/sleep."""
    for cx in (107, 149):
        draw.arc([cx - 16, 107, cx + 16, 123], start=200, end=340,
                 fill=OUTLINE, width=3)


def _draw_glasses(draw: ImageDraw.Draw) -> None:
    """Layer 9: Round glasses frames over the eyes."""
    # Left lens
    draw.ellipse([88, 95, 126, 127], outline=GLASSES, width=3)
    # Right lens
    draw.ellipse([130, 95, 168, 127], outline=GLASSES, width=3)
    # Bridge
    draw.line([(126, 111), (130, 111)], fill=GLASSES, width=2)
    # Left arm
    draw.line([(88, 111), (68, 104)], fill=GLASSES, width=2)
    # Right arm
    draw.line([(168, 111), (188, 104)], fill=GLASSES, width=2)


def _draw_eyebrows(draw: ImageDraw.Draw) -> None:
    """Layer 10: Thin arched eyebrows above the glasses."""
    # Left brow — two line segments forming a gentle arch
    draw.line([(90, 92), (106, 87), (122, 92)], fill=BROW, width=3)
    # Right brow
    draw.line([(134, 92), (150, 87), (166, 92)], fill=BROW, width=3)


def _draw_nose(draw: ImageDraw.Draw) -> None:
    """Layer 11: Tiny subtle nose curve."""
    draw.arc([122, 120, 134, 128], start=30, end=150,
             fill=MOUTH_LINE, width=1)


def _draw_blush(img: Image.Image, strong: bool = False) -> None:
    """Layer 13: Semi-transparent blush marks on cheeks.

    Args:
        strong: If True, uses stronger alpha (for sleep state).
    """
    overlay = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    ov_draw = ImageDraw.Draw(overlay)
    colour = BLUSH_STRONG if strong else BLUSH
    ov_draw.ellipse([84, 124, 114, 136], fill=colour)
    ov_draw.ellipse([142, 124, 172, 136], fill=colour)
    img.alpha_composite(overlay)


def _draw_body(draw: ImageDraw.Draw) -> None:
    """Layer 14: Neck, dark top, lab coat collar/body."""
    # Neck
    draw.rectangle([112, 162, 144, 185], fill=SKIN)
    # Dark top (neckline)
    draw.polygon([(100, 182), (156, 182), (170, 230), (86, 230)],
                 fill=TOP_DARK)
    # Lab coat shadow fold left
    draw.polygon([(60, 185), (85, 182), (80, 220), (55, 225)],
                 fill=COAT_SHADOW)
    # Lab coat shadow fold right
    draw.polygon([(196, 185), (171, 182), (176, 220), (201, 225)],
                 fill=COAT_SHADOW)
    # Lab coat left collar
    draw.polygon([(60, 185), (115, 182), (90, 256), (30, 256)],
                 fill=COAT_WHITE, outline=COAT_SHADOW)
    # Lab coat right collar
    draw.polygon([(141, 182), (196, 185), (226, 256), (166, 256)],
                 fill=COAT_WHITE, outline=COAT_SHADOW)


# ── Mouth Styles ────────────────────────────────────────────────────

def _draw_mouth_smirk(draw: ImageDraw.Draw) -> None:
    """Neutral smirk — slight curve up on the right (idle)."""
    draw.arc([114, 133, 142, 142], start=200, end=345,
             fill=MOUTH_LINE, width=2)


def _draw_mouth_flat(draw: ImageDraw.Draw) -> None:
    """Nearly flat closed mouth (blink, wake)."""
    draw.arc([116, 134, 140, 141], start=210, end=330,
             fill=MOUTH_LINE, width=2)


def _draw_mouth_open_small(draw: ImageDraw.Draw) -> None:
    """Small open oval mouth (talk_1)."""
    draw.ellipse([116, 133, 140, 146], fill=MOUTH_LINE, outline=OUTLINE, width=1)
    draw.ellipse([119, 135, 137, 144], fill=MOUTH_OPEN)


def _draw_mouth_open_wide(draw: ImageDraw.Draw) -> None:
    """Wide open mouth mid-sentence (talk_2)."""
    draw.ellipse([112, 131, 144, 149], fill=MOUTH_LINE, outline=OUTLINE, width=1)
    draw.ellipse([115, 133, 141, 147], fill=MOUTH_OPEN)
    # Teeth
    draw.line([(122, 133), (122, 136)], fill=EYE_WHITE, width=1)
    draw.line([(128, 133), (128, 136)], fill=EYE_WHITE, width=1)


def _draw_mouth_sleep(draw: ImageDraw.Draw) -> None:
    """Slightly open sleeping mouth."""
    draw.arc([118, 136, 138, 145], start=210, end=330,
             fill=MOUTH_LINE, width=2)


def _draw_zzz(draw: ImageDraw.Draw) -> None:
    """Draw ZZZ near the top-right of the head for sleep state."""
    try:
        font_s = ImageFont.truetype("arial.ttf", 10)
        font_m = ImageFont.truetype("arial.ttf", 11)
        font_l = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font_s = ImageFont.load_default()
        font_m = font_s
        font_l = font_s

    draw.text((162, 62), "z", fill=ZZZ_COLOUR, font=font_s)
    draw.text((170, 52), "z", fill=ZZZ_COLOUR, font=font_m)
    draw.text((180, 44), "z", fill=ZZZ_COLOUR, font=font_l)


# ── Shared base drawing (everything except eyes + mouth) ────────────

def _draw_base(img: Image.Image) -> ImageDraw.Draw:
    """Draw all shared layers (1-7, 9-11, 14) and return the draw context.

    Args:
        img: The RGBA image to draw on.

    Returns:
        The ImageDraw.Draw object for further drawing.
    """
    draw = ImageDraw.Draw(img)

    # Layer 1: Hair back
    _draw_hair_back(draw)
    # Layer 14: Body/coat (drawn early so hair overlaps it)
    _draw_body(draw)
    # Layer 2: Head/face
    _draw_head(img, draw)
    # Need a fresh draw after alpha_composite in _draw_head
    draw = ImageDraw.Draw(img)
    # Layer 3: Bangs
    _draw_bangs(draw)
    # Layer 4: Ahoge
    _draw_ahoge(draw)
    # Layer 5: Headphone band
    _draw_headphone_band(draw)
    # Layer 6: Ears
    _draw_ears(draw)
    # Layer 7: Headphone cups
    _draw_headphone_cups(draw)
    # Layer 10: Eyebrows
    _draw_eyebrows(draw)
    # Layer 11: Nose
    _draw_nose(draw)

    return draw


# ── Sprite generators ───────────────────────────────────────────────

def generate_idle() -> Image.Image:
    """aria_idle.png — Open eyes + neutral smirk."""
    img = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    draw = _draw_base(img)
    _draw_eyes_open(draw)
    _draw_glasses(draw)
    _draw_mouth_smirk(draw)
    _draw_blush(img)
    return img


def generate_blink() -> Image.Image:
    """aria_blink.png — Closed eyes + flat mouth."""
    img = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    draw = _draw_base(img)
    _draw_eyes_closed(draw)
    _draw_glasses(draw)
    _draw_mouth_flat(draw)
    _draw_blush(img)
    return img


def generate_wake() -> Image.Image:
    """aria_wake.png — Wide eyes + flat mouth (listening/surprised)."""
    img = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    draw = _draw_base(img)
    _draw_eyes_wide(draw)
    _draw_glasses(draw)
    _draw_mouth_flat(draw)
    _draw_blush(img)
    return img


def generate_talk_1() -> Image.Image:
    """aria_talk_1.png — Open eyes + small open mouth."""
    img = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    draw = _draw_base(img)
    _draw_eyes_open(draw)
    _draw_glasses(draw)
    _draw_mouth_open_small(draw)
    _draw_blush(img)
    return img


def generate_talk_2() -> Image.Image:
    """aria_talk_2.png — Open eyes + wide open mouth."""
    img = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    draw = _draw_base(img)
    _draw_eyes_open(draw)
    _draw_glasses(draw)
    _draw_mouth_open_wide(draw)
    _draw_blush(img)
    return img


def generate_sleep() -> Image.Image:
    """aria_sleep.png — Closed eyes + sleeping mouth + ZZZ + strong blush."""
    img = Image.new("RGBA", (SIZE, SIZE), TRANSPARENT)
    draw = _draw_base(img)
    _draw_eyes_closed(draw)
    _draw_glasses(draw)
    _draw_mouth_sleep(draw)
    _draw_blush(img, strong=True)
    # Need fresh draw after alpha_composite in _draw_blush
    draw = ImageDraw.Draw(img)
    _draw_zzz(draw)
    return img


# ── Main ────────────────────────────────────────────────────────────

def main():
    """Generate all 6 sprites and save to assets/sprites/."""
    os.makedirs(SPRITE_DIR, exist_ok=True)

    sprites = {
        "aria_idle.png": generate_idle,
        "aria_blink.png": generate_blink,
        "aria_wake.png": generate_wake,
        "aria_talk_1.png": generate_talk_1,
        "aria_talk_2.png": generate_talk_2,
        "aria_sleep.png": generate_sleep,
    }

    print("Generating Kurisu Makise chibi sprites...")
    print()

    for filename, generator in sprites.items():
        path = os.path.join(SPRITE_DIR, filename)
        img = generator()
        img.save(path, "PNG")
        size_kb = os.path.getsize(path) / 1024
        print(f"  [OK] {filename:20s} — {size_kb:.1f} KB")

    print()
    print(f"All sprites saved to {SPRITE_DIR}")


if __name__ == "__main__":
    main()
