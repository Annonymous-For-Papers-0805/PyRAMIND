"""LLM entity extraction (Innovation H) — cached via the AzureClient cache.

When `enable_entity_augmentation` is OFF, callers should skip this module
entirely and embed the raw content.
"""

from __future__ import annotations

import json

_ENTITY_KEYS = ("dates", "numbers", "names", "places", "topics")


def _empty_entities() -> dict:
    return {k: [] for k in _ENTITY_KEYS}


def extract_entities(text: str, azure_client) -> dict:
    """Extract entities from text via the Azure client. Returns a dict with keys:
    dates, numbers, names, places, topics. Always returns all keys, possibly empty.

    Errors (network, JSON parse) yield an empty result rather than raising.
    """
    prompt = f'''Extract key entities from this text. Return JSON only.

Text: "{text[:1000]}"

Return: {{"dates": ["any dates or time references"], "numbers": ["any quantities, amounts, durations, money"], "names": ["any person names"], "places": ["any locations or places"], "topics": ["2-3 word topic labels for what this text is about"]}}

If a category has nothing, return empty list. Be thorough — capture $amounts, durations like "2 days", dates like "March 15th", etc.'''

    try:
        raw = azure_client.complete(
            prompt=prompt,
            system="Extract entities from text. Return only valid JSON.",
            category="completion",
            temperature=0,
            max_tokens=300,
        )
    except Exception:
        return _empty_entities()

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        entities = json.loads(cleaned)
    except json.JSONDecodeError:
        return _empty_entities()

    if not isinstance(entities, dict):
        return _empty_entities()

    out = _empty_entities()
    for key in _ENTITY_KEYS:
        val = entities.get(key, [])
        if isinstance(val, list):
            out[key] = [str(x) for x in val]
    return out


def augment_text(content: str, entities: dict) -> str:
    """Prepend extracted entities as tags to improve embedding findability.

    Format: "WHEN: ... | QUANTITY: ... | WHO: ... | WHERE: ... | ABOUT: ... || <content>"
    Empty categories are omitted. If no entities at all, returns content unchanged.
    """
    tags = []
    if entities.get("dates"):
        tags.append(f"WHEN: {', '.join(entities['dates'])}")
    if entities.get("numbers"):
        tags.append(f"QUANTITY: {', '.join(entities['numbers'])}")
    if entities.get("names"):
        tags.append(f"WHO: {', '.join(entities['names'])}")
    if entities.get("places"):
        tags.append(f"WHERE: {', '.join(entities['places'])}")
    if entities.get("topics"):
        tags.append(f"ABOUT: {', '.join(entities['topics'])}")

    if not tags:
        return content
    return " | ".join(tags) + " || " + content
