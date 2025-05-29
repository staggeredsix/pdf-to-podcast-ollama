"""Utilities for cleaning conversation data before validation."""
from __future__ import annotations

from typing import Dict, List


def fix_conversation_json(conversation_json: Dict) -> Dict:
    """Ensure required fields exist in conversation JSON.

    This function normalizes the conversation data returned by the LLM so that
    it can be parsed by ``Conversation``.

    * If the key ``dialogues`` is present instead of ``dialogue``, it will be
      renamed.
    * Any dialogue entry missing a ``speaker`` field will be assigned by
      alternating between ``speaker-1`` and ``speaker-2``.
    """
    if "dialogue" not in conversation_json and "dialogues" in conversation_json:
        conversation_json["dialogue"] = conversation_json.pop("dialogues")

    dialogue: List[Dict] = conversation_json.get("dialogue", [])
    last = "speaker-2"
    for entry in dialogue:
        if "speaker" not in entry:
            last = "speaker-1" if last == "speaker-2" else "speaker-2"
            entry["speaker"] = last
        else:
            last = entry["speaker"]

    conversation_json["dialogue"] = dialogue
    return conversation_json
