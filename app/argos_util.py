# app/argos_util.py
from argostranslate import translate
from functools import lru_cache

@lru_cache(maxsize=8)
def _get_translation_obj(src_code: str, tgt_code: str):
    langs = translate.get_installed_languages()
    src = next((l for l in langs if getattr(l, "code", None) == src_code), None)
    tgt = next((l for l in langs if getattr(l, "code", None) == tgt_code), None)
    if not src or not tgt:
        return None

    # Tercihen yeni API
    try:
        tr = src.get_translation(tgt)  # type: ignore[attr-defined]
        if tr:
            return tr
    except Exception:
        pass

    # Eski API varyantları
    for attr in ("translations_to", "translations"):
        tlist = getattr(src, attr, None)
        if tlist:
            for t in tlist:
                to_obj = getattr(t, "to_language", None) or getattr(t, "to_lang", None)
                if to_obj and getattr(to_obj, "code", None) == tgt_code:
                    return t
    return None

def translate_offline(text: str, src: str, tgt: str) -> str:
    if not text:
        return text
    tr = _get_translation_obj(src, tgt)
    if tr is None:
        # paket yoksa metni geri döndür (sessiz degradasyon)
        return text
    try:
        return tr.translate(text)
    except Exception:
        return text
