# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for the old ``tulip.security`` import path.

``tulip.security`` used to hold two things; they now live in two places:

- The domain-neutral grounding and verification layer (``Evidence``,
  ``Severity``, ``ground_finding``, ``verify``, the taxonomy enums, …) moved to
  :mod:`tulip.control`, next to the admission gate and audit trail that use it.
  Importing one of those names from here still works, with a
  :class:`~tulip.core.warnings.TulipDeprecationWarning` naming the new path.
- The security-domain tooling (``red_team``, ``Target``, ``SecurityContext``,
  the SOC analyst, the intel / SIEM / EDR / scanner / fingerprint / AWS
  adapters, the IR playbooks, …) moved to the separate, opt-in
  ``tulip-agents-security`` distribution, imported as :mod:`tulip_security`.
  Importing one of those names from here resolves it lazily from that package
  when it is installed, and raises :class:`ImportError` saying what to install
  when it is not.

Submodule imports (``tulip.security.policy``, ``tulip.security.redteam.probes``,
…) are aliased the same way, to the very same module objects, so patch targets
and ``isinstance`` checks written against the old paths keep working.

New code should import from :mod:`tulip.control` or :mod:`tulip_security`.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
import warnings
from typing import TYPE_CHECKING, Any

from tulip.core.warnings import TulipDeprecationWarning


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


#: Names that moved to :mod:`tulip.control`.
_CORE_NAMES = frozenset(
    {
        "Abstention",
        "AdversarialSkeptic",
        "AtlasTechnique",
        "Confidence",
        "Evidence",
        "EvidenceQualitySkeptic",
        "FingerprintClassifier",
        "FingerprintFinding",
        "FingerprintVerdict",
        "GroundedFinding",
        "Indicator",
        "IndicatorType",
        "OwaspASI",
        "OwaspLLM",
        "Refutation",
        "SEVERITY_ORDER",
        "Severity",
        "Skeptic",
        "TaxonomyTag",
        "VerificationResult",
        "ground_finding",
        "ground_fingerprint",
        "is_finding",
        "severity_at_least",
        "verify",
    }
)

#: Names that moved to the ``tulip-agents-security`` distribution.
_DOMAIN_NAMES = frozenset(
    {
        "ActionsPort",
        "CloudSource",
        "DirectPromptInjection",
        "EndpointSource",
        "ExcessiveAgency",
        "FEATURE_KEYS",
        "IdentitySource",
        "IndirectPromptInjection",
        "Jailbreak",
        "LogSource",
        "PostureEvidence",
        "PostureFinding",
        "PostureReport",
        "Probe",
        "ProbeOutcome",
        "READONLY_PREFIXES",
        "SecurityAdapter",
        "SecurityContext",
        "SecurityControls",
        "Sender",
        "SensitiveInformationDisclosure",
        "Target",
        "ThreatIntelSource",
        "ToolAdapter",
        "UnsandboxedCodeExecution",
        "all_playbooks",
        "all_probes",
        "as_json",
        "assure",
        "aws_services",
        "classify_indicator",
        "cloud_posture_audit",
        "create_soc_analyst",
        "default_classifier",
        "describe_aws",
        "describe_aws_tool",
        "dispatch_timing_probe_reference",
        "enrich_indicator",
        "enrich_indicator_tool",
        "enrich_to_finding",
        "env",
        "fetch_host_timeline",
        "fetch_host_timeline_tool",
        "fingerprint_endpoint_tool",
        "fingerprint_to_finding",
        "ground_report",
        "guardrail_coverage",
        "indicator_type",
        "inference_claim",
        "is_readonly_operation",
        "isolate_host",
        "isolate_host_tool",
        "list_detections",
        "list_detections_tool",
        "measure_endpoint_timing",
        "monitor",
        "nist_800_61_ir",
        "phishing_triage",
        "query_siem",
        "ransomware_containment",
        "red_team",
        "scan_dependencies",
        "scan_dependencies_tool",
        "scan_endpoint",
        "scan_endpoint_to_finding",
        "scan_endpoint_tool",
        "security_toolset",
        "siem_query_tool",
        "submit_posture",
        "suite_probes",
        "tool_match",
        "use_aws",
        "use_aws_tool",
    }
)

#: Old submodules of the grounding layer and where each one lives now.
_CORE_MODULES = {
    "admit": "tulip.control.admission",
    "audit": "tulip.control.audit",
    "findings": "tulip.control.findings",
    "grounded": "tulip.control.grounded",
    "grounding_eval": "tulip.reasoning.grounding_eval",
    "policy": "tulip.control.policy",
    "secure": "tulip.control.governed",
    "taxonomy": "tulip.control.taxonomy",
    "verify": "tulip.control.verification",
}

#: Old submodules that now live under :mod:`tulip_security` (same names).
_DOMAIN_MODULES = frozenset(
    {
        "_adapters",
        "adapter",
        "assess",
        "aws",
        "context",
        "edr",
        "fingerprint",
        "intel",
        "jobs",
        "playbooks",
        "redteam",
        "scanner",
        "siem",
        "soc",
        "target",
        "testing",
    }
)

_DOMAIN_PACKAGE = "tulip_security"
_INSTALL_HINT = (
    "{what} moved to the separate tulip-agents-security package, which is not "
    "published to PyPI. Install it from the repository with "
    '`pip install "git+https://github.com/tuliplabs-ai/tulip-agents'
    '#subdirectory=packages/tulip-agents-security"` '
    "and import it from `tulip_security`."
)


def _moved(old: str, new: str) -> None:
    warnings.warn(
        f"{old} moved to {new}; import it from there. "
        "The tulip.security path is deprecated and will be removed in Tulip 3.0.",
        TulipDeprecationWarning,
        stacklevel=3,
    )


def _import_domain(module: str, what: str) -> ModuleType:
    """Import ``module`` from the security distribution, or say how to get it."""
    try:
        installed = importlib.util.find_spec(_DOMAIN_PACKAGE) is not None
    except ValueError:  # pragma: no cover - a module with no __spec__ in sys.modules
        installed = True
    if not installed:
        raise ImportError(_INSTALL_HINT.format(what=what), name=_DOMAIN_PACKAGE)
    return importlib.import_module(module)


def _alias_target(fullname: str) -> str | None:
    """The module an old ``tulip.security.*`` name now resolves to, if any."""
    head, _, rest = fullname.removeprefix(f"{__name__}.").partition(".")
    if head in _CORE_MODULES:
        return f"{_CORE_MODULES[head]}.{rest}" if rest else _CORE_MODULES[head]
    if head in _DOMAIN_MODULES:
        return f"{_DOMAIN_PACKAGE}.{head}.{rest}" if rest else f"{_DOMAIN_PACKAGE}.{head}"
    return None


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``tulip.security.<sub>`` to the module that replaced it.

    The loader hands back the already-imported target module itself, so the old
    and new names are one object in ``sys.modules`` — never a second copy.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if not fullname.startswith(f"{__name__}."):
            return None
        new = _alias_target(fullname)
        if new is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, origin=new)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        new = spec.origin
        if new is None:  # pragma: no cover - find_spec always sets it
            raise ImportError(spec.name)
        if new.startswith(f"{_DOMAIN_PACKAGE}."):
            module = _import_domain(new, f"`{spec.name}`")
        else:
            _moved(f"`{spec.name}`", f"`{new}`")
            module = importlib.import_module(new)
        # The import machinery stamps the alias spec onto whatever this returns;
        # keep the real one so exec_module can put it back.
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        alias_spec = module.__spec__
        assert alias_spec is not None  # set by the import machinery before exec
        module.__spec__ = alias_spec.loader_state


# Matched by name, not isinstance, so a reload of this module doesn't add a
# second finder next to the first one's (now stale) class.
if not any(
    type(finder).__module__ == __name__ and type(finder).__name__ == _AliasFinder.__name__
    for finder in sys.meta_path
):
    sys.meta_path.insert(0, _AliasFinder())


class _ShimModule(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        # ``import tulip.security.verify`` binds that submodule onto this
        # package, which would shadow the ``verify`` function of the same name.
        if name in _CORE_NAMES and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ShimModule


def __getattr__(name: str) -> Any:
    if name in _CORE_NAMES:
        _moved(f"`tulip.security.{name}`", f"`tulip.control.{name}`")
        value = getattr(importlib.import_module("tulip.control"), name)
    elif name in _DOMAIN_NAMES:
        value = getattr(_import_domain(_DOMAIN_PACKAGE, f"`tulip.security.{name}`"), name)
    elif name in _CORE_MODULES or name in _DOMAIN_MODULES:
        return importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Resolve each name once: ``from tulip.security import X`` looks X up twice,
    # and one warning per name is enough.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(_CORE_NAMES | _DOMAIN_NAMES)


__all__ = sorted(_CORE_NAMES | _DOMAIN_NAMES)  # noqa: PLE0605
