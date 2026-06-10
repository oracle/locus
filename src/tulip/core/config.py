# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Configuration management - 100% Pydantic Settings."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """Settings for model providers."""

    model_config = SettingsConfigDict(
        env_prefix="TULIP_MODEL_",
        env_file=".env",
        extra="ignore",
    )

    # Default provider and model
    default_provider: str = "openai"
    default_model: str = "gpt-4o"

    # API Keys (from environment)
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Generation defaults
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


class AgentSettings(BaseSettings):
    """Settings for agent behavior."""

    model_config = SettingsConfigDict(
        env_prefix="TULIP_AGENT_",
        env_file=".env",
        extra="ignore",
    )

    # Iteration limits
    max_iterations: int = 20
    tool_loop_threshold: int = 3

    # Reflexion
    enable_reflexion: bool = True
    confidence_threshold: float = 0.85
    diminishing_returns: bool = True

    # Grounding
    enable_grounding: bool = True
    grounding_threshold: float = 0.65
    max_replans: int = 2

    # Terminal tools
    terminal_tools: list[str] = Field(
        default_factory=lambda: ["submit", "done", "finish", "complete"]
    )


class TelemetrySettings(BaseSettings):
    """Settings for observability."""

    model_config = SettingsConfigDict(
        env_prefix="TULIP_TELEMETRY_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = False
    service_name: str = "tulip"
    otlp_endpoint: str | None = None
    otlp_headers: dict[str, str] = Field(default_factory=dict)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "text"


class CheckpointerSettings(BaseSettings):
    """Settings for state persistence."""

    model_config = SettingsConfigDict(
        env_prefix="TULIP_CHECKPOINT_",
        env_file=".env",
        extra="ignore",
    )

    backend: Literal["memory", "file", "redis", "http"] = "memory"

    # File backend
    file_path: str = ".tulip/checkpoints"

    # Redis backend
    redis_url: str | None = None

    # HTTP backend
    http_url: str | None = None
    http_headers: dict[str, str] = Field(default_factory=dict)

    # Delta storage
    enable_delta: bool = True
    delta_chain_limit: int = 5


class TulipSettings(BaseSettings):
    """Root settings for Tulip SDK."""

    model_config = SettingsConfigDict(
        env_prefix="TULIP_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Environment
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Nested settings
    model: ModelSettings = Field(default_factory=ModelSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    checkpointer: CheckpointerSettings = Field(default_factory=CheckpointerSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TulipSettings:
        """Create settings from a dictionary."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Export settings to dictionary."""
        return self.model_dump()


# Global settings instance (lazy loaded)
_settings: TulipSettings | None = None


def get_settings() -> TulipSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = TulipSettings()
    return _settings


def configure(settings: TulipSettings | dict[str, Any] | None = None) -> TulipSettings:
    """
    Configure global settings.

    Args:
        settings: Settings instance or dict to configure with

    Returns:
        Configured settings
    """
    global _settings
    if settings is None:
        _settings = TulipSettings()
    elif isinstance(settings, dict):
        _settings = TulipSettings.from_dict(settings)
    else:
        _settings = settings
    return _settings
