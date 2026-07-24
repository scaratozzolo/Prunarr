"""
Unit tests for JustWatch GraphQL request batching.

Covers the aliased batched search (:meth:`JustWatchClient.search_titles_batch`),
the ``nodes(ids:)`` batched offers lookup
(:meth:`JustWatchClient.get_offers_batch`), and the streaming-cache prewarm
(:meth:`StreamingChecker.prewarm`) that ties them together.
"""

import pytest

from prunarr.justwatch.client import JustWatchClient
from prunarr.services.streaming_checker import StreamingChecker


class FakeResponse:
    """Minimal ``requests.Response`` stand-in for a successful GraphQL call."""

    status_code = 200

    def __init__(self, data):
        self._data = data
        self.text = "body"

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def build_fake_post(search_db, offers_db, calls):
    """Return a fake ``requests.post`` that answers batched search/offers queries."""

    def fake_post(url, json=None, headers=None, timeout=None):
        query, variables = json["query"], json["variables"]

        if "BatchSearch" in query:
            calls.append("search")
            data = {}
            index = 0
            while f"f{index}" in variables:
                title = variables[f"f{index}"]["searchQuery"]
                node_id = search_db.get(title)
                if node_id is None:
                    data[f"t{index}"] = {"edges": []}
                else:
                    data[f"t{index}"] = {
                        "edges": [
                            {
                                "node": {
                                    "id": node_id,
                                    "objectType": "MOVIE",
                                    "content": {
                                        "title": title,
                                        "originalReleaseYear": 2000,
                                        "externalIds": {"imdbId": "tt0", "tmdbId": 1},
                                    },
                                }
                            }
                        ]
                    }
                index += 1
            return FakeResponse({"data": data})

        if "GetOffersBatch" in query:
            calls.append("offers")
            nodes = [{"id": nid, "offers": offers_db.get(nid, [])} for nid in variables["ids"]]
            return FakeResponse({"data": {"nodes": nodes}})

        return FakeResponse({"data": {}})

    return fake_post


def flatrate_offer(technical_name):
    """A FLATRATE offer package for the given provider."""
    return {
        "monetizationType": "FLATRATE",
        "presentationType": "HD",
        "package": {
            "packageId": 8,
            "shortName": technical_name[:3],
            "clearName": technical_name.title(),
            "technicalName": technical_name,
        },
    }


class FakeCache:
    """In-memory cache manager stand-in."""

    def __init__(self):
        self.store = {}

    def is_enabled(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


@pytest.fixture
def client():
    return JustWatchClient(locale="en_US")


class TestSearchTitlesBatch:
    def test_chunks_requests_and_aligns_results(self, client, mocker):
        calls = []
        mocker.patch(
            "requests.post", side_effect=build_fake_post({"The Matrix": "tm10"}, {}, calls)
        )

        results = client.search_titles_batch([("The Matrix", 1999)] * 30, content_type="MOVIE")

        assert calls.count("search") == 2  # 30 / SEARCH_BATCH_SIZE(25) -> 2 requests
        assert len(results) == 30
        assert results[0] is not None and results[0].id == "tm10"

    def test_unmatched_title_is_none(self, client, mocker):
        calls = []
        mocker.patch("requests.post", side_effect=build_fake_post({}, {}, calls))

        results = client.search_titles_batch([("Nonexistent", 2001)], content_type="MOVIE")

        assert results == [None]


class TestGetOffersBatch:
    def test_chunks_requests(self, client, mocker):
        calls = []
        mocker.patch("requests.post", side_effect=build_fake_post({}, {}, calls))

        client.get_offers_batch([f"id{i}" for i in range(25)], providers=["netflix"])

        assert calls.count("offers") == 3  # 25 / OFFERS_BATCH_SIZE(10) -> 3 requests

    def test_filters_to_flatrate_on_configured_providers(self, client, mocker):
        offers_db = {"tmA": [flatrate_offer("netflix")], "tmB": [flatrate_offer("amazon")]}
        calls = []
        mocker.patch("requests.post", side_effect=build_fake_post({}, offers_db, calls))

        result = client.get_offers_batch(["tmA", "tmB"], providers=["netflix"])

        assert [o.provider_short_name for o in result["tmA"]] == ["net"]
        assert result["tmB"] == []  # amazon filtered out

    def test_deduplicates_ids(self, client, mocker):
        calls = []
        mocker.patch("requests.post", side_effect=build_fake_post({}, {}, calls))

        client.get_offers_batch(["dup", "dup", "dup"], providers=None)

        assert calls.count("offers") == 1


class TestPrewarm:
    def test_populates_cache_with_minimal_requests(self, mocker):
        search_db = {"The Matrix": "tmA", "Obscure": "tmB"}  # "Nonexistent" absent
        offers_db = {"tmA": [flatrate_offer("netflix")], "tmB": [flatrate_offer("amazon")]}
        calls = []
        mocker.patch("requests.post", side_effect=build_fake_post(search_db, offers_db, calls))

        cache = FakeCache()
        cache.set("streaming_movie_tt4", True)  # pre-cached -> must be skipped
        checker = StreamingChecker(locale="en_US", providers=["netflix"], cache_manager=cache)

        entries = [
            {"title": "The Matrix", "year": 1999, "id": "tt1"},  # netflix -> True
            {"title": "Obscure", "year": 2000, "id": "tt2"},  # amazon only -> False
            {"title": "Nonexistent", "year": 2001, "id": "tt3"},  # no match -> False
            {"title": "Cached", "year": 2002, "id": "tt4"},  # already cached -> skip
        ]
        resolved = checker.prewarm(entries, "movie")

        assert resolved == 3  # tt4 skipped
        assert cache.store["streaming_movie_tt1"] is True
        assert cache.store["streaming_movie_tt2"] is False
        assert cache.store["streaming_movie_tt3"] is False
        assert cache.store["streaming_movie_tt4"] is True  # untouched
        assert calls == ["search", "offers"]  # one batched request per stage

    def test_noop_when_all_cached(self, mocker):
        calls = []
        mocker.patch("requests.post", side_effect=build_fake_post({}, {}, calls))
        cache = FakeCache()
        cache.set("streaming_movie_tt1", True)
        checker = StreamingChecker(locale="en_US", providers=["netflix"], cache_manager=cache)

        resolved = checker.prewarm([{"title": "X", "year": 1, "id": "tt1"}], "movie")

        assert resolved == 0
        assert calls == []
