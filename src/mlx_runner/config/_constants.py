# Per-field coercion metadata: name -> (python base type, is_optional).
# Used to parse ``config set KEY VALUE`` strings and to validate loaded JSON.
_FIELD_SPEC = {
    "model": (str, True),
    "system": (str, True),
    "max_tokens": (int, False),
    "temperature": (float, False),
    "top_p": (float, False),
    "top_k": (int, False),
    "min_p": (float, False),
    "repetition_penalty": (float, True),
    "seed": (int, True),
    "max_kv_size": (int, True),
    "kv_bits": (int, True),
    "safety_fraction": (float, False),
}

_NULL_TOKENS = {"", "none", "null", "default", "unset"}
