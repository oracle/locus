# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for core config module."""

from locus.core.config import (
    AgentSettings,
    CheckpointerSettings,
    LocusSettings,
    ModelSettings,
    TelemetrySettings,
    configure,
    get_settings,
)


class TestModelSettings:
    """Tests for ModelSettings."""

    def test_default_settings(self):
        """Test default model settings."""
        settings = ModelSettings()
        assert settings.default_provider == "openai"
        assert settings.default_model == "gpt-4o"
        assert settings.max_tokens == 4096
        assert settings.temperature == 0.7
        assert settings.top_p == 0.9

    def test_oci_defaults(self):
        """Test OCI default settings."""
        settings = ModelSettings()
        assert settings.oci_profile == "DEFAULT"
        assert settings.oci_auth_type == "security_token"
        assert settings.oci_region == "us-chicago-1"
        assert settings.oci_compartment_id is None

    def test_openai_api_key_none_by_default(self, monkeypatch):
        """Test API key is None by default."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = ModelSettings()
        assert settings.openai_api_key is None


class TestAgentSettings:
    """Tests for AgentSettings."""

    def test_default_settings(self):
        """Test default agent settings."""
        settings = AgentSettings()
        assert settings.max_iterations == 20
        assert settings.tool_loop_threshold == 3
        assert settings.enable_reflexion is True
        assert settings.confidence_threshold == 0.85
        assert settings.diminishing_returns is True

    def test_grounding_defaults(self):
        """Test grounding default settings."""
        settings = AgentSettings()
        assert settings.enable_grounding is True
        assert settings.grounding_threshold == 0.65
        assert settings.max_replans == 2

    def test_terminal_tools_default(self):
        """Test default terminal tools."""
        settings = AgentSettings()
        assert "submit" in settings.terminal_tools
        assert "done" in settings.terminal_tools
        assert "finish" in settings.terminal_tools
        assert "complete" in settings.terminal_tools


class TestTelemetrySettings:
    """Tests for TelemetrySettings."""

    def test_default_settings(self):
        """Test default telemetry settings."""
        settings = TelemetrySettings()
        assert settings.enabled is False
        assert settings.service_name == "locus"
        assert settings.otlp_endpoint is None
        assert settings.otlp_headers == {}

    def test_logging_defaults(self):
        """Test logging default settings."""
        settings = TelemetrySettings()
        assert settings.log_level == "INFO"
        assert settings.log_format == "text"


class TestCheckpointerSettings:
    """Tests for CheckpointerSettings."""

    def test_default_settings(self):
        """Test default checkpointer settings."""
        settings = CheckpointerSettings()
        assert settings.backend == "memory"
        assert settings.file_path == ".locus/checkpoints"
        assert settings.redis_url is None
        assert settings.http_url is None
        assert settings.http_headers == {}

    def test_delta_defaults(self):
        """Test delta storage defaults."""
        settings = CheckpointerSettings()
        assert settings.enable_delta is True
        assert settings.delta_chain_limit == 5


class TestLocusSettings:
    """Tests for LocusSettings."""

    def test_default_settings(self):
        """Test default root settings."""
        settings = LocusSettings()
        assert settings.env == "development"
        assert settings.debug is False

    def test_nested_settings(self):
        """Test nested settings are created."""
        settings = LocusSettings()
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.agent, AgentSettings)
        assert isinstance(settings.telemetry, TelemetrySettings)
        assert isinstance(settings.checkpointer, CheckpointerSettings)

    def test_from_dict(self):
        """Test creating settings from dictionary."""
        data = {
            "env": "production",
            "debug": True,
        }
        settings = LocusSettings.from_dict(data)
        assert settings.env == "production"
        assert settings.debug is True

    def test_from_dict_with_nested(self):
        """Test from_dict with nested settings."""
        data = {
            "env": "staging",
            "model": {
                "default_provider": "openai",
                "temperature": 0.5,
            },
        }
        settings = LocusSettings.from_dict(data)
        assert settings.env == "staging"
        assert settings.model.default_provider == "openai"
        assert settings.model.temperature == 0.5

    def test_to_dict(self):
        """Test exporting settings to dictionary."""
        settings = LocusSettings()
        data = settings.to_dict()

        assert isinstance(data, dict)
        assert data["env"] == "development"
        assert "model" in data
        assert "agent" in data
        assert "telemetry" in data
        assert "checkpointer" in data


class TestGetSettings:
    """Tests for get_settings function."""

    def setup_method(self):
        """Reset global settings before each test."""
        import locus.core.config as config_module

        config_module._settings = None

    def test_get_settings_creates_default(self):
        """Test get_settings creates default settings."""
        import locus.core.config as config_module

        config_module._settings = None

        settings = get_settings()

        assert settings is not None
        assert isinstance(settings, LocusSettings)

    def test_get_settings_returns_same_instance(self):
        """Test get_settings returns same instance."""
        import locus.core.config as config_module

        config_module._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2


class TestConfigure:
    """Tests for configure function."""

    def setup_method(self):
        """Reset global settings before each test."""
        import locus.core.config as config_module

        config_module._settings = None

    def test_configure_none_creates_default(self):
        """Test configure with None creates default."""
        settings = configure(None)

        assert isinstance(settings, LocusSettings)
        assert settings.env == "development"

    def test_configure_with_dict(self):
        """Test configure with dictionary."""
        settings = configure({"env": "production", "debug": True})

        assert settings.env == "production"
        assert settings.debug is True

    def test_configure_with_settings_instance(self):
        """Test configure with settings instance."""
        custom = LocusSettings(env="staging")
        settings = configure(custom)

        assert settings is custom
        assert settings.env == "staging"

    def test_configure_updates_global(self):
        """Test configure updates global settings."""
        import locus.core.config as config_module

        configure({"env": "production"})

        assert config_module._settings is not None
        assert config_module._settings.env == "production"
