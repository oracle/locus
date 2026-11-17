# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The control-layer conformance fixtures under ``conformance/v1`` are current.

Two kinds of check. The first regenerates the fixtures into a temporary
directory and fails on any difference from the committed tree: a format change
is a deliberate ``conformance/v2``, never a silent edit of ``v1``. The rest
read the committed files back through the SDK with no help from the export
script's expectations: every trail verifies with the committed report, every
policy hashes, lints and decides as committed, every digest and id matches,
every sequence and admission replays to the committed result. A hand edit of
a fixture fails here with a message that names the file.

The last test runs a dependency-free Node script over the vectors when
``node`` is on the path: the digests and signatures are recomputed from
another language, which is what the fixtures exist for.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest

from locus.control import (
    Ed25519Signer,
    InMemoryApprovals,
    PolicyDocument,
    PolicyFileError,
    call_digest,
    check_jsonl,
    policy_hash,
)
from locus.control.policy_lint import lint_policy


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "conformance" / "v1"
SCRIPT = ROOT / "scripts" / "conformance_export.py"
NODE_CHECK = FIXTURES / "tools" / "check.mjs"

pytest.importorskip("cryptography")
pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def export() -> ModuleType:
    spec = importlib.util.spec_from_file_location("conformance_export", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_keys(ids: list[str] | None) -> dict[str, bytes] | None:
    if ids is None:
        return None
    return {
        key_id: (FIXTURES / "audit" / "keys" / f"{key_id}.pub.pem").read_bytes() for key_id in ids
    }


# --------------------------------------------------------------------------- drift


def test_committed_fixtures_match_regeneration(tmp_path: Path, export: ModuleType) -> None:
    export.generate(tmp_path)
    differences = export.tree_diff(FIXTURES, tmp_path)
    detail = ""
    for line in differences:
        kind, _, relative = line.partition(": ")
        if kind != "changed":
            continue
        before, after = FIXTURES / relative, tmp_path / relative
        try:
            detail = "\n".join(
                difflib.unified_diff(
                    before.read_text(encoding="utf-8").splitlines(),
                    after.read_text(encoding="utf-8").splitlines(),
                    f"committed/{relative}",
                    f"regenerated/{relative}",
                    lineterm="",
                    n=1,
                )
            )
        except UnicodeDecodeError:
            detail = f"{relative} differs (binary)"
        break
    assert not differences, (
        "conformance/v1 differs from what scripts/conformance_export.py generates; "
        "a format change is a new conformance version, not an edit of v1:\n"
        + "\n".join(differences)
        + ("\n\n" + detail if detail else "")
    )


def test_hand_written_files_are_present() -> None:
    readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
    for name in ("canonical-json", "audit", "policies", "approvals", "admission", "tools"):
        assert f"`{name}/`" in readme, f"README does not describe {name}/"
        assert (FIXTURES / name).is_dir()
    assert NODE_CHECK.is_file()


# --------------------------------------------------------------------------- canonical JSON


def test_canonical_json_vectors() -> None:
    vectors = _json(FIXTURES / "canonical-json" / "vectors.json")
    assert vectors
    for vector in vectors:
        ensure_ascii = vector["mode"] == "ascii"
        for source in (vector["input"], json.loads(vector["input_json"])):
            canonical = json.dumps(
                source,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=ensure_ascii,
                default=str,
            )
            assert canonical == vector["canonical"], vector["name"]
        digest = hashlib.sha256(vector["canonical"].encode("utf-8")).hexdigest()
        assert digest == vector["sha256"], vector["name"]


def test_vectors_tie_to_the_other_directories() -> None:
    by_name = {v["name"]: v for v in _json(FIXTURES / "canonical-json" / "vectors.json")}
    first = (FIXTURES / "audit" / "unsigned" / "trail.jsonl").read_text().splitlines()[0]
    assert by_name["audit-entry-body"]["sha256"] == json.loads(first)["hash"]
    assert (
        by_name["policy-default"]["sha256"]
        == (FIXTURES / "policies" / "default.hash").read_text().strip()
    )
    digests = _json(FIXTURES / "approvals" / "digests.json")
    assert by_name["approval-call"]["sha256"] == digests[0]["digest"]


# --------------------------------------------------------------------------- audit


def _audit_checks() -> list[tuple[str, Path, dict[str, Any]]]:
    found: list[tuple[str, Path, dict[str, Any]]] = []
    for name in ("unsigned", "signed", "rotated"):
        found.append(
            (
                name,
                FIXTURES / "audit" / name / "trail.jsonl",
                _json(FIXTURES / "audit" / name / "check.json"),
            )
        )
    for check in sorted((FIXTURES / "audit" / "tamper").glob("*.check.json")):
        stem = check.name[: -len(".check.json")]
        found.append((f"tamper/{stem}", check.with_name(f"{stem}.jsonl"), _json(check)))
    return found


@pytest.mark.parametrize(
    ("name", "trail", "spec"),
    _audit_checks(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_audit_trails_check_as_committed(name: str, trail: Path, spec: dict[str, Any]) -> None:
    text = trail.read_text(encoding="utf-8")
    assert spec["checks"], name
    for check in spec["checks"]:
        report = check_jsonl(
            text, keys=_public_keys(check["keys"]), expected_head=check["expected_head"]
        )
        assert asdict(report) == check["report"], (
            f"{name} with keys={check['keys']} head={check['expected_head']}"
        )


@pytest.mark.parametrize("name", ["unsigned", "signed", "rotated"])
def test_expected_head_is_the_last_record(name: str) -> None:
    folder = FIXTURES / "audit" / name
    head = (folder / "expected_head").read_text().strip()
    lines = (folder / "trail.jsonl").read_text().splitlines()
    assert head == json.loads(lines[-1])["hash"]
    assert head == _json(folder / "check.json")["expected_head"]
    assert len(lines) == 4


def test_fixture_keys_and_derived_key_id() -> None:
    keys = FIXTURES / "audit" / "keys"
    private = sorted(keys.glob("*.priv.pem"))
    assert len(private) == 2
    derived = 0
    for path in private:
        key_id = path.name[: -len(".priv.pem")]
        signer = Ed25519Signer.from_pem(path.read_bytes())
        assert signer.public_key_pem() == (keys / f"{key_id}.pub.pem").read_bytes()
        if signer.key_id == key_id:
            derived += 1
            from cryptography.hazmat.primitives import serialization

            public = serialization.load_pem_public_key((keys / f"{key_id}.pub.pem").read_bytes())
            raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            assert hashlib.sha256(raw).hexdigest()[:16] == key_id
        else:
            assert key_id == "conformance-v1-rotated"
    assert derived == 1


def test_signed_records_carry_the_record_fields_only() -> None:
    for name, fields in (
        ("unsigned", ["seq", "ts", "event_type", "payload", "prev_hash", "hash"]),
        (
            "signed",
            ["seq", "ts", "event_type", "payload", "prev_hash", "hash", "key_id", "signature"],
        ),
    ):
        for line in (FIXTURES / "audit" / name / "trail.jsonl").read_text().splitlines():
            assert list(json.loads(line)) == fields, name


# --------------------------------------------------------------------------- policies


def _policy_documents() -> list[Path]:
    return sorted((FIXTURES / "policies").glob("*.policy.*"))


@pytest.mark.parametrize("path", _policy_documents(), ids=lambda path: path.name)
def test_policy_documents_hash_lint_and_decide_as_committed(path: Path, export: ModuleType) -> None:
    name = path.name.split(".")[0]
    fmt: Literal["json", "yaml"] = "json" if path.suffix == ".json" else "yaml"
    document = PolicyDocument.load(path)
    assert document.dump(format=fmt) == path.read_text(encoding="utf-8"), (
        f"{name}: the SDK writes different text"
    )
    assert policy_hash(document.policy) == (path.parent / f"{name}.hash").read_text().strip()
    report = lint_policy(document.policy, environments=document.environments, kinds=document.kinds)
    assert report.to_dict() == _json(path.parent / f"{name}.lint.json")
    cases = _json(path.parent / f"{name}.decisions.json")
    assert cases
    for case in cases:
        assert export.decide(document.policy, case) == case["expected"], f"{name}: {case['action']}"


@pytest.mark.parametrize(
    "path",
    sorted((FIXTURES / "policies" / "malformed").glob("*.policy.*")),
    ids=lambda path: path.name,
)
def test_malformed_policies_fail_on_the_committed_key(path: Path, export: ModuleType) -> None:
    name = path.name.split(".")[0]
    expected = _json(path.parent / f"{name}.error.json")
    fmt: Literal["json", "yaml"] = "json" if path.suffix == ".json" else "yaml"
    with pytest.raises(PolicyFileError) as info:
        PolicyDocument.loads(path.read_text(encoding="utf-8"), format=fmt, source="")
    assert info.value.key == expected["key"]
    assert export.malformed_error(path.read_text(encoding="utf-8"), fmt) == expected


# --------------------------------------------------------------------------- approvals


def test_approval_digests_and_ids() -> None:
    cases = _json(FIXTURES / "approvals" / "digests.json")
    assert cases
    for case in cases:
        digest = call_digest(case["principal"], case["tool"], case["arguments"], case["context"])
        assert digest == case["digest"], case
        assert digest.startswith(case["approval_id"][len("appr-") : -len("-1")])
        store = InMemoryApprovals()
        assert (
            store.submit(
                case["principal"], case["tool"], case["arguments"], context=case["context"]
            )
            == case["approval_id"]
        )
    assert cases[0]["digest"] == cases[1]["digest"] == cases[2]["digest"]
    assert len({case["digest"] for case in cases[3:]}) == len(cases[3:])


@pytest.mark.parametrize(
    "path",
    sorted((FIXTURES / "approvals" / "sequences").glob("*.json")),
    ids=lambda path: path.stem,
)
def test_approval_sequences_replay_to_the_committed_record(path: Path, export: ModuleType) -> None:
    spec = _json(path)
    replayed = export.replay_sequence(spec)
    assert replayed["results"] == spec["results"]
    assert replayed["final"] == spec["final"]


# --------------------------------------------------------------------------- admission


def test_admission_cases_write_the_committed_line(export: ModuleType) -> None:
    policies = export.load_policies(FIXTURES / "policies")
    cases = _json(FIXTURES / "admission" / "cases.json")
    assert cases
    for case in cases:
        assert export.run_admission_case(case, policies) == case["expected"], case["name"]
        record = json.loads(case["expected"]["line"])
        assert record["event_type"] == "action-admission"
        committed_hash = (FIXTURES / "policies" / f"{case['policy']}.hash").read_text().strip()
        assert record["payload"]["policy_hash"] == committed_hash, case["name"]
        assert check_jsonl(case["expected"]["line"]).ok, case["name"]


# --------------------------------------------------------------------------- another language


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_recomputes_digests_and_verifies_signatures() -> None:
    node = shutil.which("node")
    assert node is not None
    completed = subprocess.run(
        [node, str(NODE_CHECK), str(FIXTURES)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["vectors"] == len(_json(FIXTURES / "canonical-json" / "vectors.json"))
    signed = sum(
        len((FIXTURES / "audit" / name / "trail.jsonl").read_text().splitlines())
        for name in ("signed", "rotated")
    )
    assert result["signatures"] == signed
    derived = result["derived_key_ids"]
    assert set(derived) == {"2ee9031729dbde22", "conformance-v1-rotated"}
    assert derived["2ee9031729dbde22"] == "2ee9031729dbde22"
    assert derived["conformance-v1-rotated"] != "conformance-v1-rotated"
