# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Azure OpenAI.

Azure serves OpenAI's models, but not at OpenAI's address and not with OpenAI's
auth — which is why it cannot be one more row in the OpenAI-compatible routing
table the way Groq or DeepSeek can. Three things differ:

* the URL carries a **deployment name**, not a model id, and deployments are
  named by whoever created them — ``gpt4o-prod`` is a perfectly ordinary name
  for ``gpt-4o``;
* auth is an ``api-key`` header, not ``Authorization: Bearer``;
* every request needs an ``api-version`` query parameter.

The ``openai`` package already knows all of this through ``AsyncAzureOpenAI``,
so this provider is :class:`~tulip.models.native.openai.OpenAIModel` with a
different client behind it. Everything downstream — message conversion, tool
calls, streaming, structured output — is inherited unchanged, which is the
point: Azure should not be a second implementation that drifts from the first.

    from tulip import Agent, AgentConfig

    # AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY from the environment
    agent = Agent(config=AgentConfig(model="azure:gpt4o-prod"))

    # or explicitly
    agent = Agent(config=AgentConfig(model=get_model(
        "azure:gpt4o-prod",
        endpoint="https://my-resource.openai.azure.com",
        api_version="2024-10-21",
    )))

Uses the same ``openai`` extra as the OpenAI provider — no new dependency.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tulip.core.loop_bound import loop_bound
from tulip.models.native.openai import OpenAIConfig, OpenAIModel


if TYPE_CHECKING:  # pragma: no cover - typing only
    import openai


#: Azure pins behaviour to a dated API version rather than shipping a rolling
#: "latest". This is a GA version that supports tool calling and streaming;
#: override it when a deployment needs a newer feature.
DEFAULT_API_VERSION = "2024-10-21"


class AzureOpenAIConfig(OpenAIConfig):
    """Configuration for Azure-hosted OpenAI deployments."""

    endpoint: str | None = Field(
        default=None,
        description="Resource endpoint, e.g. https://my-resource.openai.azure.com",
    )
    api_version: str = Field(
        default=DEFAULT_API_VERSION,
        description="Azure API version. Azure has no rolling 'latest'.",
    )
    azure_ad_token: str | None = Field(
        default=None,
        description="Entra ID (Azure AD) token, used instead of an API key.",
    )


class AzureOpenAIModel(OpenAIModel):
    """Azure OpenAI provider.

    Example:
        >>> model = AzureOpenAIModel(model="gpt4o-prod")  # a *deployment* name
        >>> response = await model.complete([Message.user("Hello!")])
    """

    config: AzureOpenAIConfig

    def __init__(
        self,
        model: str,
        endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        config = AzureOpenAIConfig(
            model=model,
            endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or DEFAULT_API_VERSION,
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            **kwargs,
        )
        # Skip OpenAIModel.__init__, which would rebuild a plain OpenAIConfig
        # and drop the Azure fields.
        super(OpenAIModel, self).__init__(config=config)

    @property
    def client(self) -> openai.AsyncOpenAI:
        """An ``AsyncAzureOpenAI``, which is an ``AsyncOpenAI`` subclass.

        Loop-bound for the same reason as the OpenAI client: it wraps an httpx
        pool, and one cached across two event loops fails on the second with
        ``APIConnectionError`` — which reads as a provider outage rather than
        the caching bug it is.
        """

        def build() -> openai.AsyncOpenAI:
            import openai  # noqa: PLC0415

            if not self.config.endpoint:
                raise ValueError(
                    "Azure OpenAI needs a resource endpoint. Set "
                    "AZURE_OPENAI_ENDPOINT, or pass endpoint=... — for example "
                    "https://my-resource.openai.azure.com"
                )
            if not (self.config.api_key or self.config.azure_ad_token):
                raise ValueError(
                    "No credentials for Azure OpenAI. Set AZURE_OPENAI_API_KEY, "
                    "or pass api_key=... / azure_ad_token=..."
                )
            return openai.AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                azure_ad_token=self.config.azure_ad_token,
                api_version=self.config.api_version,
                max_retries=self.config.max_retries,
                timeout=self.config.request_timeout,
            )

        return loop_bound(self, "_client", build)
