"""
Unit tests for the JustWatch client's rate-limiting behavior.

Covers request throttling and the 429/403 exponential-backoff retry logic added
to :class:`prunarr.justwatch.client.JustWatchClient`.
"""

import pytest
import requests

from prunarr.justwatch.client import JustWatchClient
from prunarr.justwatch.exceptions import JustWatchAPIError, JustWatchRateLimitError


class FakeResponse:
    """Minimal stand-in for a ``requests.Response``."""

    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data if data is not None else {"data": {"ok": True}}
        self.text = "body"

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(str(self.status_code))
            error.response = self
            raise error


@pytest.fixture(autouse=True)
def reset_throttle():
    """Reset the shared throttle clock before each test for isolation."""
    JustWatchClient._last_request_ts = 0.0
    yield
    JustWatchClient._last_request_ts = 0.0


@pytest.fixture
def client():
    """A JustWatch client with no logger/cache."""
    return JustWatchClient(locale="en_US")


def _backoff_sleeps(sleep_mock):
    """Sleep durations that are backoffs (>= INITIAL_BACKOFF), not throttle waits."""
    return [
        call.args[0]
        for call in sleep_mock.call_args_list
        if call.args and call.args[0] >= JustWatchClient.INITIAL_BACKOFF
    ]


class TestRateLimiting:
    """429/403 backoff-retry behavior."""

    def test_successful_request_is_not_retried(self, client, mocker):
        post = mocker.patch("requests.post", return_value=FakeResponse(200, {"data": {"x": 1}}))
        mocker.patch("time.sleep")

        assert client._make_request("q", {}) == {"x": 1}
        assert post.call_count == 1

    def test_403_then_success_retries_with_exponential_backoff(self, client, mocker):
        post = mocker.patch(
            "requests.post",
            side_effect=[
                FakeResponse(403),
                FakeResponse(403),
                FakeResponse(200, {"data": {"x": 1}}),
            ],
        )
        sleep = mocker.patch("time.sleep")

        assert client._make_request("q", {}) == {"x": 1}
        assert post.call_count == 3
        assert _backoff_sleeps(sleep) == [5.0, 10.0]

    def test_persistent_403_raises_rate_limit_error(self, client, mocker):
        post = mocker.patch("requests.post", return_value=FakeResponse(403))
        mocker.patch("time.sleep")

        with pytest.raises(JustWatchRateLimitError):
            client._make_request("q", {})
        assert post.call_count == JustWatchClient.MAX_RETRIES

    def test_persistent_429_raises_rate_limit_error(self, client, mocker):
        post = mocker.patch("requests.post", return_value=FakeResponse(429))
        mocker.patch("time.sleep")

        with pytest.raises(JustWatchRateLimitError):
            client._make_request("q", {})
        assert post.call_count == JustWatchClient.MAX_RETRIES

    def test_server_error_is_not_treated_as_rate_limit(self, client, mocker):
        post = mocker.patch("requests.post", return_value=FakeResponse(500))
        mocker.patch("time.sleep")

        with pytest.raises(JustWatchAPIError):
            client._make_request("q", {})
        assert post.call_count == 1  # fail fast, no retry


class TestThrottle:
    """Minimum-interval spacing between requests."""

    def test_enforces_minimum_interval_between_requests(self, client, mocker):
        mocker.patch("requests.post", return_value=FakeResponse(200, {"data": {}}))
        sleep = mocker.patch("time.sleep")

        client._make_request("q", {})  # first call: no wait needed
        client._make_request("q", {})  # second call: must wait ~MIN_REQUEST_INTERVAL

        throttle_waits = [
            call.args[0]
            for call in sleep.call_args_list
            if call.args and 0 < call.args[0] <= JustWatchClient.MIN_REQUEST_INTERVAL
        ]
        assert throttle_waits, "expected a throttle sleep between consecutive requests"
