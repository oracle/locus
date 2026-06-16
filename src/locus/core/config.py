# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Configuration management - 100% Pydantic Settings."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """Settings for model providers."""

    model_config = SettingsConfigDict(
        env_prefix="LOCUS_MODEL_",
        env_file=".env",
        extra="ignore",
    )

    # Default provider and model
    default_provider: str = "openai"
    default_model: str = "gpt-4o"

    # API Keys (from environment)
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")

    # OCI Settings
    oci_profile: str = "DEFAULT"
    oci_auth_type: Literal["security_token", "api_key", "instance_principal"] = "security_token"
    oci_compartment_id: str | None = None
    oci_region: str = "us-chicago-1"

    # Generation defaults
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


class AgentSettings(BaseSettings):
    """Settings for agent behavior."""

    model_config = SettingsConfigDict(
        env_prefix="LOCUS_AGENT_",
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
        env_prefix="LOCUS_TELEMETRY_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = False
    service_name: str = "locus"
    otlp_endpoint: str | None = None
    otlp_headers: dict[str, str] = Field(default_factory=dict)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "text"


class CheckpointerSettings(BaseSettings):
    """Settings for state persistence."""

    model_config = SettingsConfigDict(
        env_prefix="LOCUS_CHECKPOINT_",
        env_file=".env",
        extra="ignore",
    )

    backend: Literal["memory", "file", "redis", "http"] = "memory"

    # File backend
    file_path: str = ".locus/checkpoints"

    # Redis backend
    redis_url: str | None = None

    # HTTP backend
    http_url: str | None = None
    http_headers: dict[str, str] = Field(default_factory=dict)

    # Delta storage
    enable_delta: bool = True
    delta_chain_limit: int = 5


class LocusSettings(BaseSettings):
    """Root settings for Locus SDK."""

    model_config = SettingsConfigDict(
        env_prefix="LOCUS_",
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
    def from_dict(cls, data: dict[str, Any]) -> LocusSettings:
        """Create settings from a dictionary."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Export settings to dictionary."""
        return self.model_dump()


# Global settings instance (lazy loaded)
_settings: LocusSettings | None = None


def get_settings() -> LocusSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = LocusSettings()
    return _settings


def configure(settings: LocusSettings | dict[str, Any] | None = None) -> LocusSettings:
    """
    Configure global settings.

    Args:
        settings: Settings instance or dict to configure with

    Returns:
        Configured settings
    """
    global _settings
    if settings is None:
        _settings = LocusSettings()
    elif isinstance(settings, dict):
        _settings = LocusSettings.from_dict(settings)
    else:
        _settings = settings
    return _settings
