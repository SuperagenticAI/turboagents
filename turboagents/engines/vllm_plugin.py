"""Experimental vLLM plugin entry point for TurboAgents.

The plugin is intentionally conservative. It exposes a load hook through the
`vllm.general_plugins` entry-point group so vLLM can discover TurboAgents at
runtime via `VLLM_PLUGINS=turboagents`.

The actual kernel/backend integration is still future work; today this plugin
mainly establishes a stable integration seam and leaves breadcrumbs in the
process environment for future worker/backend code.
"""

from __future__ import annotations

import logging
import os


LOGGER = logging.getLogger(__name__)


def register() -> None:
    bits = os.environ.get("TURBOAGENTS_VLLM_BITS", "3.5")
    mode = os.environ.get("TURBOAGENTS_VLLM_MODE", "safe")
    LOGGER.info(
        "TurboAgents vLLM plugin loaded (experimental). bits=%s mode=%s",
        bits,
        mode,
    )
