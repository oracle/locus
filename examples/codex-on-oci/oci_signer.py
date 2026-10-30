# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Sign the gateway's OCI Generative AI calls with an OCI CLI profile.

LiteLLM's OCI provider signs a request with API-key fields given one by one,
or with an ``oci_signer`` object. A YAML file cannot carry an object, so this
callback builds one from the profile named by ``OCI_PROFILE`` in
``~/.oci/config`` (``OCI_CONFIG_FILE`` names another file) and attaches it to
every OCI call after the router has chosen the deployment and before the
request leaves (``async_pre_call_deployment_hook``). Retries and fallbacks go
through it too.

An API-key profile is read once. A session-token profile, the kind
``oci session authenticate`` writes, is re-read at most once a minute, so a
refreshed session is picked up without a restart. The key and the token stay
on this host; Codex only ever sees the loopback gateway.

Two request fixes ride along. OCI's chat API refuses function tools together
with ``reasoning_effort`` on the GPT-5 family, and Codex sends both on every
turn, so the effort is dropped when tools are present (set to ``none`` for
GPT-6 ids, which OCI serves with reasoning on by default). And when
``OCI_COMPARTMENT_ID`` is unset, the profile's tenancy is used.

Wire it in ``config.yaml``::

    litellm_settings:
      callbacks: [oci_signer.handler]
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from litellm.integrations.custom_logger import CustomLogger


PROFILE = os.environ.get("OCI_PROFILE", "DEFAULT")
CONFIG_FILE = os.environ.get("OCI_CONFIG_FILE", "~/.oci/config")
REFRESH_SECONDS = 60


class OCIProfileSigner(CustomLogger):
    """Attach an OCI request signer, built from a CLI profile, to each OCI call."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._signer: Any = None
        self._loaded_at = 0.0
        self._tenancy = ""

    def _load(self) -> None:
        import oci  # noqa: PLC0415 - the OCI SDK is only needed once a call is made

        config = oci.config.from_file(file_location=CONFIG_FILE, profile_name=PROFILE)
        self._tenancy = str(config.get("tenancy", ""))
        token_file = config.get("security_token_file")
        if token_file:
            token = Path(token_file).expanduser().read_text().strip()
            key = oci.signer.load_private_key_from_file(
                str(Path(config["key_file"]).expanduser()), config.get("pass_phrase")
            )
            self._signer = oci.auth.signers.SecurityTokenSigner(token, key)
        else:
            self._signer = oci.signer.Signer.from_config(config)
        self._loaded_at = time.monotonic()

    def signer(self) -> Any:
        """The signer for the profile, rebuilt for a session profile once a minute."""
        with self._lock:
            session = self._signer is not None and hasattr(self._signer, "token")
            stale = time.monotonic() - self._loaded_at > REFRESH_SECONDS
            if self._signer is None or (session and stale):
                self._load()
            return self._signer

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, Any], call_type: Any
    ) -> dict[str, Any] | None:
        model = str(kwargs.get("model") or "")
        if not (model.startswith("oci/") or kwargs.get("custom_llm_provider") == "oci"):
            return kwargs
        kwargs["oci_signer"] = self.signer()
        if not kwargs.get("oci_compartment_id"):
            kwargs["oci_compartment_id"] = os.environ.get("OCI_COMPARTMENT_ID") or self._tenancy
        optional = kwargs.get("optional_params")
        tools = kwargs.get("tools") or (optional or {}).get("tools")
        if tools:
            effort = "none" if "openai.gpt-6" in model else None
            for params in (kwargs, optional if isinstance(optional, dict) else None):
                if params is None:
                    continue
                params.pop("reasoning_effort", None)
                if effort:
                    params["reasoning_effort"] = effort
        return kwargs


handler = OCIProfileSigner()
