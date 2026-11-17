# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The policy as a document: YAML and JSON round trips, strict keys, hash, hot reload.

A policy file has to say exactly what the Python API says, no more and no
less: a key the policy does not have is an error that names it, and a dumped
policy loads back to one that decides every action the same way. The watcher
swaps the policy between admissions and never leaves the gate without one,
and every trail entry names the rules it was decided under.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from locus.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    PolicyDocument,
    PolicyFileError,
    PolicyWatcher,
    Severity,
    admit,
    admit_sync,
    approve,
    check_jsonl,
    gate_tool,
    policy_hash,
)
from locus.control.policy_file import SCHEMA, dump_policy, load_policy, loads_policy, to_dict
from locus.control.verification import VerificationResult
from locus.tools.decorator import tool


if TYPE_CHECKING:
    from collections.abc import Callable


_FULL = ControlPolicy(
    require_verification_score=0.6,
    max_blast_radius=3,
    require_human_for=frozenset({"production", "iam"}),
    deny_for=frozenset({"prohibited"}),
    min_severity=Severity.MEDIUM,
    require_sandbox_for=frozenset({"code"}),
    version="full-2026-11",
    require_human_over_usd=250.0,
    spend_limit_usd=5_000.0,
)

_OPEN = ControlPolicy(
    require_verification_score=0.0,
    max_blast_radius=1,
    require_human_for=frozenset({"high_value"}),
)

_STRONG = VerificationResult(survives=True, confidence=0.95, evidence_quality=0.9)

#: Actions that between them reach every rule of a policy.
_BATTERY = [
    Action(name="noop"),
    Action(name="refund", environment="production", kind="payment"),
    Action(name="refund", environment="staging", kind="payment", blast_radius=5),
    Action(name="grant", environment="staging", kind="iam"),
    Action(name="wipe", environment="production", tags=frozenset({"prohibited"})),
    Action(name="run", kind="code"),
    Action(name="run", kind="code", tags=frozenset({"sandboxed"})),
    Action(name="pay", kind="payment", cost_usd=300.0),
    Action(name="pay", kind="payment", cost_usd=4_900.0),
    Action(name="pay", kind="payment", cost_usd=20.0, tags=frozenset({"high_value"})),
]


def _decisions(policy: ControlPolicy) -> list[tuple[str, list[str]]]:
    out = []
    for action in _BATTERY:
        for verdict in (None, _STRONG):
            decision = approve(action, policy=policy, verdict=verdict, spent_usd=200.0)
            out.append((decision.outcome, decision.checks))
    return out


# --------------------------------------------------------------------- round trips


@pytest.mark.parametrize("policy", [ControlPolicy(), _FULL, _OPEN])
@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_dump_then_load_is_the_same_policy_and_decides_the_same(
    policy: ControlPolicy, fmt: str
) -> None:
    text = policy.dump(format=fmt)  # type: ignore[arg-type]

    loaded = ControlPolicy.load(text)

    assert loaded == policy
    assert _decisions(loaded) == _decisions(policy)
    assert loaded.content_hash() == policy.content_hash()


def test_the_yaml_form_reads_as_the_reviewer_expects() -> None:
    text = _FULL.dump()

    assert text.splitlines()[0] == f'schema: "{SCHEMA}"'
    assert 'require_human_for:\n  - "iam"\n  - "production"\n' in text  # sorted labels
    assert 'min_severity: "medium"' in text
    assert "spend_limit_usd: 5000.0" in text
    assert 'deny_for:\n  - "prohibited"' in text


def test_a_json_dump_is_plain_json_with_every_rule() -> None:
    data = json.loads(_OPEN.dump(format="json"))

    assert data["schema"] == SCHEMA
    assert data["require_human_for"] == ["high_value"]
    assert data["require_human_over_usd"] is None
    assert set(data) == {"schema", *to_dict(_OPEN)}


def test_dump_writes_the_file_in_the_format_its_suffix_names(tmp_path: Path) -> None:
    yaml_path = tmp_path / "policy.yaml"
    json_path = tmp_path / "policy.json"

    _FULL.dump(yaml_path)
    _FULL.dump(json_path)

    assert yaml_path.read_text().startswith("schema:")
    assert json.loads(json_path.read_text())["version"] == "full-2026-11"
    assert ControlPolicy.load(yaml_path) == _FULL
    assert ControlPolicy.load(str(json_path)) == _FULL


def test_a_string_with_a_newline_is_a_document_and_one_without_is_a_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "p.yml"
    path.write_text("max_blast_radius: 7\n")

    assert ControlPolicy.load(str(path)).max_blast_radius == 7
    assert ControlPolicy.load("max_blast_radius: 8\n").max_blast_radius == 8
    assert ControlPolicy.load('{"max_blast_radius": 9}').max_blast_radius == 9
    assert ControlPolicy.loads("max_blast_radius: 10").max_blast_radius == 10
    with pytest.raises(FileNotFoundError):
        ControlPolicy.load("max_blast_radius: 11")


def test_omitted_keys_take_the_python_defaults() -> None:
    assert ControlPolicy.load("version: only\n") == ControlPolicy(version="only")
    assert ControlPolicy.load("{}") == ControlPolicy()


def test_an_exponent_float_survives_yaml() -> None:
    policy = ControlPolicy(spend_limit_usd=1e-5, require_human_over_usd=1e21)

    assert ControlPolicy.load(policy.dump()) == policy


def test_module_functions_match_the_methods(tmp_path: Path) -> None:
    path = tmp_path / "p.json"

    assert loads_policy(dump_policy(_OPEN, path)) == load_policy(path) == _OPEN
    assert policy_hash(_OPEN) == _OPEN.content_hash()


# --------------------------------------------------------------------- strictness


@pytest.mark.parametrize(
    ("text", "key", "fragment"),
    [
        ("max_blast_radious: 2\n", "max_blast_radious", "did you mean 'max_blast_radius'"),
        ('{"rules": {}}', "rules", "unknown key 'rules'"),
        ('{"require_human_for": ["a", 3]}', "require_human_for[1]", "non-empty string"),
        ('{"require_human_for": [""]}', "require_human_for[0]", "non-empty string"),
        ('{"require_human_for": "production"}', "require_human_for", "list of labels"),
        ('{"deny_for": {"a": 1}}', "deny_for", "list of labels"),
        ('{"max_blast_radius": 1.5}', "max_blast_radius", "expected an integer"),
        ('{"max_blast_radius": true}', "max_blast_radius", "expected an integer"),
        ('{"max_blast_radius": -1}', "max_blast_radius", "0 or more"),
        ('{"require_verification_score": "high"}', "require_verification_score", "number"),
        ('{"require_verification_score": -0.1}', "require_verification_score", "0 or more"),
        ('{"spend_limit_usd": "5000"}', "spend_limit_usd", "expected a number"),
        ('{"min_severity": "urgent"}', "min_severity", "expected one of"),
        ('{"min_severity": 3}', "min_severity", "expected one of"),
        ('{"version": 3}', "version", "expected a string"),
        ('{"schema": "locus.control.policy/v0"}', "schema", "expected 'locus.control.policy/v1'"),
        ('{"governs": {"kind": []}}', "governs.kind", "did you mean 'kinds'"),
        ('{"governs": {"environments": [1]}}', "governs.environments[0]", "non-empty string"),
        ('{"governs": []}', "governs", "expected a mapping"),
        ("[]", "", "expected a mapping of rules, got a list"),
        ("- a\n- b\n", "", "expected a mapping of rules"),
    ],
)
def test_bad_documents_name_the_key(text: str, key: str, fragment: str) -> None:
    with pytest.raises(PolicyFileError) as exc:
        ControlPolicy.load(text)

    assert exc.value.key == key
    assert fragment in str(exc.value)
    if key:
        assert f"at {key}:" in str(exc.value)


def test_the_file_name_is_in_the_error(tmp_path: Path) -> None:
    path = tmp_path / "refunds.yaml"
    path.write_text("deny_for: production\n")

    with pytest.raises(PolicyFileError, match=r"refunds\.yaml: at deny_for: expected a list"):
        ControlPolicy.load(path)


def test_unparseable_text_is_a_policy_file_error() -> None:
    with pytest.raises(PolicyFileError, match="not valid JSON"):
        ControlPolicy.load('{"max_blast_radius": ')
    with pytest.raises(PolicyFileError, match="not valid YAML"):
        ControlPolicy.load("a: [\nb: }\n")


def test_non_finite_numbers_are_refused_both_ways() -> None:
    with pytest.raises(PolicyFileError, match="finite"):
        ControlPolicy.load("spend_limit_usd: .inf\n")
    with pytest.raises(ValueError, match="cannot write"):
        ControlPolicy(spend_limit_usd=float("inf")).dump()


def test_yaml_reading_without_pyyaml_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def no_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "yaml", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_yaml)

    with pytest.raises(ImportError, match="pip install pyyaml"):
        ControlPolicy.load("max_blast_radius: 2\n")
    # JSON and YAML writing need nothing.
    assert ControlPolicy.load('{"max_blast_radius": 2}').max_blast_radius == 2
    assert "max_blast_radius: 2" in ControlPolicy(max_blast_radius=2).dump()


# --------------------------------------------------------------------- documents


def test_a_document_carries_what_the_policy_governs_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(
        _OPEN.dump() + 'governs:\n  environments: ["staging", "production"]\n  kinds: ["payment"]\n'
    )

    document = PolicyDocument.load(path)

    assert document.policy == _OPEN
    assert document.environments == {"production", "staging"}
    assert document.kinds == {"payment"}
    assert document.source == str(path)
    again = PolicyDocument.loads(document.dump())
    assert (again.policy, again.environments, again.kinds) == (
        _OPEN,
        document.environments,
        document.kinds,
    )
    assert "governs" not in PolicyDocument(policy=_OPEN).dump()
    assert ControlPolicy.load(path) == _OPEN  # the policy ignores the metadata


def test_a_document_dumps_as_json_too() -> None:
    document = PolicyDocument(policy=_OPEN, kinds=frozenset({"payment"}))

    data = json.loads(document.dump(format="json"))

    assert data["governs"] == {"kinds": ["payment"]}


# --------------------------------------------------------------------- the hash


def test_equal_policies_hash_the_same_however_they_were_built() -> None:
    built = ControlPolicy(require_human_for=frozenset({"b", "a"}), version="v")
    loaded = ControlPolicy.load('{"require_human_for": ["a", "b"], "version": "v"}')

    assert built.content_hash() == loaded.content_hash()
    assert len(built.content_hash()) == 64


def test_any_rule_change_changes_the_hash() -> None:
    base = _FULL.content_hash()

    for changed in (
        ControlPolicy(**{**_asdict(_FULL), "version": "other"}),
        ControlPolicy(**{**_asdict(_FULL), "max_blast_radius": 4}),
        ControlPolicy(**{**_asdict(_FULL), "deny_for": frozenset()}),
        ControlPolicy(**{**_asdict(_FULL), "min_severity": Severity.HIGH}),
        ControlPolicy(**{**_asdict(_FULL), "spend_limit_usd": None}),
    ):
        assert changed.content_hash() != base


def _asdict(policy: ControlPolicy) -> dict[str, Any]:
    return {name: getattr(policy, name) for name in to_dict(policy)}


# --------------------------------------------------------------------- the trail


async def test_every_admission_entry_names_the_policy_it_was_decided_under() -> None:
    trail = AuditTrail()
    strict = ControlPolicy(require_verification_score=0.0, require_human_for=frozenset({"payment"}))

    async def perform() -> str:
        return "ok"

    await admit(Action(name="refund", kind="payment"), perform, policy=_OPEN, trail=trail)
    with pytest.raises(AdmissionError):
        await admit(Action(name="refund", kind="payment"), perform, policy=strict, trail=trail)
    admit_sync(Action(name="refund", kind="payment"), lambda: "ok", policy=_OPEN, trail=trail)

    hashes = [record.payload["policy_hash"] for record in trail.records()]
    assert hashes == [_OPEN.content_hash(), strict.content_hash(), _OPEN.content_hash()]
    assert trail.verify()
    assert check_jsonl(trail.export_jsonl()).ok


def test_a_trail_written_before_the_hash_existed_still_verifies() -> None:
    # Entries recorded by an older runtime carry no ``policy_hash``: the key
    # lives in the payload, which is free-form, so nothing about the chain
    # changes for them and a reader that wants the hash finds it absent.
    trail = AuditTrail(clock=lambda: "2026-01-01T00:00:00+00:00")
    trail.record("action-admission", {"action": "refund", "outcome": "allow", "reason": "ok"})
    exported = trail.export_jsonl()

    assert check_jsonl(exported).ok
    assert "policy_hash" not in json.loads(exported)["payload"]


# --------------------------------------------------------------------- the watcher


def _write(path: Path, text: str, *, tick: int) -> None:
    """Write and give the file a distinct mtime, however fast the test runs."""
    path.write_text(text)
    os.utime(path, ns=(1_700_000_000_000_000_000 + tick, 1_700_000_000_000_000_000 + tick))


def _clock(start: float = 0.0) -> tuple[Callable[[], float], list[float]]:
    now = [start]
    return (lambda: now[0]), now


async def test_the_watcher_swaps_the_policy_between_admissions(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    _write(path, _OPEN.dump(), tick=1)
    clock, now = _clock()
    swaps: list[tuple[str, str]] = []
    watcher = PolicyWatcher(
        path,
        interval=1.0,
        clock=clock,
        on_reload=lambda old, new: swaps.append((old.content_hash(), new.content_hash())),
    )
    trail = AuditTrail()
    action = Action(name="refund", kind="payment")

    async def perform() -> str:
        return "ok"

    assert await admit(action, perform, policy=watcher, trail=trail) == "ok"

    strict = ControlPolicy(require_verification_score=0.0, require_human_for=frozenset({"payment"}))
    _write(path, strict.dump(), tick=2)
    # Not yet time to look: the policy in force is still the one loaded.
    assert await admit(action, perform, policy=watcher, trail=trail) == "ok"
    now[0] = 1.5
    with pytest.raises(AdmissionError):
        await admit(action, perform, policy=watcher, trail=trail)

    hashes = [record.payload["policy_hash"] for record in trail.records()]
    assert hashes == [_OPEN.content_hash(), _OPEN.content_hash(), strict.content_hash()]
    assert swaps == [(_OPEN.content_hash(), strict.content_hash())]
    assert watcher.reloads == 1
    assert watcher.policy == strict
    assert watcher.hash == strict.content_hash()
    assert watcher.path == path
    assert trail.verify()


def test_a_file_that_stops_parsing_keeps_the_last_good_policy(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    _write(path, _OPEN.dump(format="json"), tick=1)
    watcher = PolicyWatcher(path, interval=0.0)

    _write(path, '{"max_blast_radious": 2}', tick=2)

    assert watcher.reload() is False
    assert watcher.current() == _OPEN
    assert isinstance(watcher.last_error, PolicyFileError)
    assert watcher.reloads == 0
    # A broken file is parsed once per change, not once per poll.
    watcher._last_error = None  # noqa: SLF001 — observe that reload() does not re-parse
    assert watcher.reload() is False
    assert watcher.last_error is None

    _write(path, _FULL.dump(format="json"), tick=3)
    assert watcher.reload() is True
    assert watcher.current() == _FULL
    assert watcher.last_error is None
    assert watcher.document.policy == _FULL


def test_a_file_that_disappears_keeps_the_last_good_policy(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    _write(path, _OPEN.dump(), tick=1)
    watcher = PolicyWatcher(path, interval=0.0)

    path.unlink()

    assert watcher.reload() is False
    assert watcher.current() == _OPEN
    assert isinstance(watcher.last_error, FileNotFoundError)

    _write(path, _OPEN.dump(), tick=2)
    assert watcher.reload() is False  # same rules again: nothing changed
    assert watcher.last_error is None


def test_a_rewrite_with_the_same_rules_is_not_a_change(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    _write(path, _OPEN.dump(), tick=1)
    calls: list[int] = []
    watcher = PolicyWatcher(path, interval=0.0, on_reload=lambda _old, _new: calls.append(1))

    _write(path, "# a comment\n" + _OPEN.dump(), tick=2)

    assert watcher.reload() is False
    assert watcher.reloads == 1
    assert calls == []


def test_a_broken_file_fails_at_construction(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("deny_for: production\n")

    with pytest.raises(PolicyFileError):
        PolicyWatcher(path)
    with pytest.raises(FileNotFoundError):
        PolicyWatcher(tmp_path / "missing.yaml")


async def test_a_gated_tool_follows_the_watcher(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    _write(path, _OPEN.dump(), tick=1)
    watcher = PolicyWatcher(path, interval=0.0)
    trail = AuditTrail()

    @tool
    def issue_refund(order_id: str) -> str:
        """Refund an order."""
        return f"refunded {order_id}"

    gated = gate_tool(
        issue_refund,
        policy=watcher,
        action=lambda name, kwargs: Action(name=name, asset=kwargs["order_id"], kind="payment"),
        trail=trail,
    )

    assert await gated.execute(order_id="o1") == "refunded o1"

    strict = ControlPolicy(require_verification_score=0.0, require_human_for=frozenset({"payment"}))
    _write(path, strict.dump(), tick=2)
    refusal = json.loads(await gated.execute(order_id="o2"))

    assert refusal["outcome"] == "require_human"
    hashes = [record.payload["policy_hash"] for record in trail.records()]
    assert hashes == [_OPEN.content_hash(), strict.content_hash()]
