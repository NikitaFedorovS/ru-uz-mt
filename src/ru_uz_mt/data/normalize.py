"""Text normalization for Russian and Uzbek (script detection, transliteration)."""

from __future__ import annotations

import re
import unicodedata

# Official Uzbek Cyrillic -> Latin mapping (1995 alphabet). Context-dependent letters
# (е, ц) are handled in :func:`uzbek_cyr_to_lat`.
_CYR2LAT: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "ғ": "gʻ", "д": "d", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "қ": "q", "л": "l",
    "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ў": "oʻ", "ф": "f", "х": "x", "ҳ": "h", "ч": "ch", "ш": "sh",
    "щ": "sh", "ъ": "ʼ", "ь": "", "э": "e", "ю": "yu", "я": "ya", "ы": "i",
}  # fmt: skip
_CYR_VOWELS = set("аеёиоуўэюяы")
_UZ_SPECIFIC_CYR = set("ўқғҳ")
# Letters that occur in Russian but not in Uzbek Cyrillic.
_RU_ONLY_CYR = set("ыщ")

OKINA = "ʻ"  # ʻ  used in oʻ / gʻ
TUTUQ = "ʼ"  # ʼ  glottal stop
_APOSTROPHES = "'`‘’ʻʼ´"
_OG_APOS_RE = re.compile(rf"([oOgG])[{_APOSTROPHES}]")
_TUTUQ_RE = re.compile(rf"(?<=\w)[{_APOSTROPHES}](?=\w)")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Unicode-normalize (NFC), drop control chars and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch in "\t\n" or unicodedata.category(ch)[0] != "C")
    return _WS_RE.sub(" ", text).strip()


def script_ratio(text: str, script: str) -> float:
    """Share of alphabetic characters that belong to ``script`` ("cyrillic" or "latin")."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    if script == "cyrillic":
        hits = sum(1 for ch in letters if "Ѐ" <= ch <= "ӿ")
    elif script == "latin":
        hits = sum(1 for ch in letters if ch.isascii() or "À" <= ch <= "ɏ")
    else:
        raise ValueError(f"Unknown script: {script}")
    return hits / len(letters)


def is_cyrillic(text: str, threshold: float = 0.5) -> bool:
    """True if most letters of ``text`` are Cyrillic."""
    return script_ratio(text, "cyrillic") > threshold


def looks_like_russian(text: str) -> bool:
    """Heuristic: Cyrillic text with Russian-only letters and no Uzbek-specific ones."""
    lower = set(text.lower())
    return bool(lower & _RU_ONLY_CYR) and not (lower & _UZ_SPECIFIC_CYR)


def _apply_case(out: str, ch: str, neighbor_upper: bool) -> str:
    if not ch.isupper():
        return out
    return out.upper() if neighbor_upper else out[:1].upper() + out[1:]


def uzbek_cyr_to_lat(text: str) -> str:
    """Transliterate Uzbek Cyrillic to the official Latin alphabet.

    Rules beyond the 1:1 table: ``е`` -> ``ye`` at word start / after a vowel or ъ/ь;
    ``ц`` -> ``s`` at word start / after a consonant, ``ts`` after a vowel.
    """
    result: list[str] = []
    for i, ch in enumerate(text):
        low = ch.lower()
        prev = text[i - 1].lower() if i > 0 else ""
        word_start = not prev.isalpha()
        if low == "е":
            out = "ye" if word_start or prev in _CYR_VOWELS or prev in "ъь" else "e"
        elif low == "ц":
            out = "ts" if prev in _CYR_VOWELS else "s"
        elif low in _CYR2LAT:
            out = _CYR2LAT[low]
        else:
            result.append(ch)
            continue
        neighbors = text[max(0, i - 1) : i] + text[i + 1 : i + 2]
        neighbor_upper = any(n.isalpha() and n.isupper() for n in neighbors)
        result.append(_apply_case(out, ch, neighbor_upper))
    return "".join(result)


def normalize_uzbek_apostrophes(text: str) -> str:
    """Unify apostrophe variants: ``o‘/g'`` -> ``oʻ/gʻ``; in-word apostrophes -> ``ʼ``."""
    text = _TUTUQ_RE.sub(TUTUQ, text)
    return _OG_APOS_RE.sub(lambda m: m.group(1) + OKINA, text)
