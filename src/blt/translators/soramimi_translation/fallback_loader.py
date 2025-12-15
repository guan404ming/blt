"""Fallback mapping loader for soramimi translation"""

import json
import logging
from importlib.resources import files

logger = logging.getLogger(__name__)


def load_mapping(target_lang: str) -> dict[str, str]:
    """
    Load fallback mapping from JSON file

    Args:
        target_lang: Target language code

    Returns:
        Fallback mapping dict (falls back to en.json if language not found)
    """
    # Normalize language code
    lang_code = target_lang.lower().split("-")[0]  # e.g., "zh-cn" -> "zh"

    # Map common variants to canonical names
    lang_map = {
        "cmn": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
    }
    lang_code = lang_map.get(lang_code, lang_code)

    # Get the fallback mappings package
    try:
        mappings_pkg = files("blt.translators.shared").joinpath("fallback_mappings")

        # Try language-specific file first
        lang_file = mappings_pkg.joinpath(f"{lang_code}.json")
        try:
            content = lang_file.read_text(encoding="utf-8")
            logger.info(f"   Loaded fallback mapping for {lang_code}")
            return json.loads(content)
        except (FileNotFoundError, IsADirectoryError):
            pass
    except Exception as e:
        logger.warning(f"Failed to load from package resources: {e}")

    # Fall back to default English mapping
    try:
        mappings_pkg = files("blt.translators.shared").joinpath("fallback_mappings")
        en_file = mappings_pkg.joinpath("en.json")
        content = en_file.read_text(encoding="utf-8")
        logger.info("   Using default fallback mapping (en.json)")
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Failed to load default fallback mapping: {e}")

    # Last resort: return empty dict
    logger.warning(f"No fallback mapping found for {target_lang}, using empty mapping")
    return {}
