from __future__ import annotations

import re


COMMODITY_VOCAB = {
    "AU": 0,
    "CU": 1,
    "NI": 2,
    "W": 3,
    "LI": 4,
}


def commodity_id(target_element: str) -> int:
    return COMMODITY_VOCAB.get(target_element.upper(), len(COMMODITY_VOCAB))


def feature_prefix(name: str) -> str:
    match = re.match(r"([A-Za-z0-9]+)", name)
    return (match.group(1) if match else name).upper()
