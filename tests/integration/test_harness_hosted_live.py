# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The hosted-sandbox backends against their services: the conformance kit.

Each test needs an account and is skipped without its key: ``E2B_API_KEY``
for :class:`E2BBackend`, ``DAYTONA_API_KEY`` for :class:`DaytonaBackend`. A
sandbox is created for the test and destroyed after it. These have not been
run by the maintainers; they are here so that whoever has an account can.
"""

from __future__ import annotations

import os

import pytest

from locus.harness.testing import assert_backend_conformance


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.environ.get("E2B_API_KEY"), reason="E2B_API_KEY is not set")
def test_the_e2b_backend_conforms() -> None:
    pytest.importorskip("e2b")
    from locus.harness import E2BBackend

    report = assert_backend_conformance(
        lambda: E2BBackend.create(template=os.environ.get("LOCUS_E2B_TEMPLATE") or None)
    )
    assert report.failed == []


@pytest.mark.skipif(not os.environ.get("DAYTONA_API_KEY"), reason="DAYTONA_API_KEY is not set")
def test_the_daytona_backend_conforms() -> None:
    pytest.importorskip("daytona")
    from locus.harness import DaytonaBackend

    report = assert_backend_conformance(
        lambda: DaytonaBackend.create(snapshot=os.environ.get("LOCUS_DAYTONA_SNAPSHOT") or None)
    )
    assert report.failed == []
