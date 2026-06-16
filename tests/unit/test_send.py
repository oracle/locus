# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for Send (map-reduce) module."""

import pytest

from locus.core.send import (
    Send,
    SendBatch,
    SendResult,
    aggregate_send_results,
    broadcast,
    extract_send_results,
    is_send,
    is_send_list,
    normalize_sends,
    scatter,
    send,
)


class TestSend:
    """Tests for Send class."""

    def test_basic_creation(self):
        """Test basic Send creation."""
        s = Send(node="worker", payload={"task": "process"})
        assert s.node == "worker"
        assert s.payload == {"task": "process"}
        assert s.send_id.startswith("send_")

    def test_frozen(self):
        """Test Send is immutable."""
        from pydantic import ValidationError

        s = Send(node="worker")
        with pytest.raises(ValidationError, match="frozen"):
            s.node = "other"

    def test_with_payload(self):
        """Test with_payload method."""
        s = Send(node="worker", payload={"a": 1})
        new_s = s.with_payload(b=2)
        assert new_s.payload == {"a": 1, "b": 2}
        assert s.payload == {"a": 1}  # Original unchanged

    def test_with_metadata(self):
        """Test with_metadata method."""
        s = Send(node="worker", metadata={"index": 0})
        new_s = s.with_metadata(total=10)
        assert new_s.metadata == {"index": 0, "total": 10}


class TestSendResult:
    """Tests for SendResult class."""

    def test_success_result(self):
        """Test successful SendResult."""
        sr = SendResult(
            send_id="send_123",
            node="worker",
            success=True,
            result={"data": "processed"},
            duration_ms=100.5,
        )
        assert sr.success
        assert sr.result == {"data": "processed"}
        assert sr.error is None

    def test_failure_result(self):
        """Test failed SendResult."""
        sr = SendResult(
            send_id="send_123",
            node="worker",
            success=False,
            error="Connection failed",
        )
        assert not sr.success
        assert sr.error == "Connection failed"
        assert sr.result is None


class TestSendBatch:
    """Tests for SendBatch class."""

    def test_creation(self):
        """Test SendBatch creation."""
        sends = [
            Send(node="worker", payload={"task": 1}),
            Send(node="worker", payload={"task": 2}),
            Send(node="analyzer", payload={"data": "x"}),
        ]
        batch = SendBatch(sends=sends, source_node="splitter")
        assert batch.count == 3
        assert "worker" in batch.target_nodes
        assert "analyzer" in batch.target_nodes

    def test_group_by_node(self):
        """Test group_by_node method."""
        sends = [
            Send(node="worker", payload={"task": 1}),
            Send(node="worker", payload={"task": 2}),
            Send(node="analyzer", payload={"data": "x"}),
        ]
        batch = SendBatch(sends=sends, source_node="splitter")
        groups = batch.group_by_node()
        assert len(groups["worker"]) == 2
        assert len(groups["analyzer"]) == 1


class TestIsSend:
    """Tests for is_send function."""

    def test_detects_send(self):
        """Test is_send with Send instance."""
        assert is_send(Send(node="worker"))

    def test_rejects_non_send(self):
        """Test is_send with non-Send values."""
        assert not is_send({})
        assert not is_send(None)
        assert not is_send({"node": "worker"})


class TestIsSendList:
    """Tests for is_send_list function."""

    def test_detects_send_list(self):
        """Test is_send_list with list of Sends."""
        assert is_send_list([Send(node="a"), Send(node="b")])

    def test_rejects_empty_list(self):
        """Test empty list is valid."""
        assert is_send_list([])

    def test_rejects_mixed_list(self):
        """Test list with non-Send elements."""
        assert not is_send_list([Send(node="a"), {"node": "b"}])

    def test_rejects_non_list(self):
        """Test non-list values."""
        assert not is_send_list(Send(node="a"))
        assert not is_send_list(None)


class TestNormalizeSends:
    """Tests for normalize_sends function."""

    def test_normalize_single_send(self):
        """Test normalizing single Send."""
        s = Send(node="worker")
        result = normalize_sends(s)
        assert result == [s]

    def test_normalize_send_list(self):
        """Test normalizing list of Sends."""
        sends = [Send(node="a"), Send(node="b")]
        result = normalize_sends(sends)
        assert result == sends

    def test_returns_none_for_non_send(self):
        """Test returns None for non-Send values."""
        assert normalize_sends({}) is None
        assert normalize_sends("string") is None
        assert normalize_sends(None) is None


class TestExtractSendResults:
    """Tests for extract_send_results function."""

    def test_extracts_successful_results(self):
        """Test extracting results from successful sends."""
        results = [
            SendResult(send_id="s1", node="worker", success=True, result={"data": 1}),
            SendResult(send_id="s2", node="worker", success=False, error="failed"),
            SendResult(send_id="s3", node="worker", success=True, result={"data": 3}),
        ]
        extracted = extract_send_results(results)
        assert extracted == {"s1": {"data": 1}, "s3": {"data": 3}}
        assert "s2" not in extracted


class TestAggregateSendResults:
    """Tests for aggregate_send_results function."""

    def test_default_aggregation(self):
        """Test default aggregation returns list."""
        results = [
            SendResult(send_id="s1", node="worker", success=True, result=1),
            SendResult(send_id="s2", node="worker", success=True, result=2),
        ]
        aggregated = aggregate_send_results(results)
        assert aggregated == [1, 2]

    def test_with_reducer(self):
        """Test aggregation with custom reducer."""
        results = [
            SendResult(send_id="s1", node="worker", success=True, result={"a": 1}),
            SendResult(send_id="s2", node="worker", success=True, result={"b": 2}),
        ]
        aggregated = aggregate_send_results(results, reducer=lambda a, b: {**a, **b})
        assert aggregated == {"a": 1, "b": 2}

    def test_filters_failures(self):
        """Test that failed results are filtered out."""
        results = [
            SendResult(send_id="s1", node="worker", success=True, result=1),
            SendResult(send_id="s2", node="worker", success=False, error="failed"),
        ]
        aggregated = aggregate_send_results(results)
        assert aggregated == [1]

    def test_all_failures_with_reducer(self):
        """Test aggregation with reducer when all results failed returns None."""
        results = [
            SendResult(send_id="s1", node="worker", success=False, error="failed 1"),
            SendResult(send_id="s2", node="worker", success=False, error="failed 2"),
        ]
        aggregated = aggregate_send_results(results, reducer=lambda a, b: a + b)
        assert aggregated is None


class TestSendConvenienceFunctions:
    """Tests for send convenience functions."""

    def test_send_function(self):
        """Test send() function."""
        s = send("worker", task="process", data=[1, 2, 3])
        assert s.node == "worker"
        assert s.payload == {"task": "process", "data": [1, 2, 3]}

    def test_broadcast_function(self):
        """Test broadcast() function."""
        sends = broadcast(["w1", "w2", "w3"], {"task": "analyze"})
        assert len(sends) == 3
        assert all(s.payload == {"task": "analyze"} for s in sends)
        assert [s.node for s in sends] == ["w1", "w2", "w3"]

    def test_scatter_function(self):
        """Test scatter() function."""
        sends = scatter("processor", [10, 20, 30])
        assert len(sends) == 3
        assert sends[0].payload == {"item": 10, "index": 0, "total": 3}
        assert sends[1].payload == {"item": 20, "index": 1, "total": 3}
        assert sends[2].payload == {"item": 30, "index": 2, "total": 3}

    def test_scatter_custom_key(self):
        """Test scatter with custom key."""
        sends = scatter("processor", ["a", "b"], key="data")
        assert sends[0].payload["data"] == "a"
        assert sends[1].payload["data"] == "b"

    def test_scatter_without_index(self):
        """Test scatter without index."""
        sends = scatter("processor", [1, 2], include_index=False)
        assert "index" not in sends[0].payload
        assert "total" not in sends[0].payload
