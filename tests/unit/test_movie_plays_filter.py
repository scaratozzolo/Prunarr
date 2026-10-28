"""Unit tests for play-count filtering on movies."""

from prunarr.commands.movies import apply_movie_filters


def _movie(title, plays):
    return {"title": title, "watch_count": plays, "watch_status": "watched", "user": "alice"}


MOVIES = [_movie("never", 0), _movie("once", 1), _movie("often", 5)]


def _titles(movies):
    return [m["title"] for m in movies]


def test_no_plays_filter_keeps_all():
    assert _titles(apply_movie_filters(MOVIES)) == ["never", "once", "often"]


def test_max_plays_zero_keeps_only_unwatched():
    assert _titles(apply_movie_filters(MOVIES, max_plays=0)) == ["never"]


def test_min_plays():
    assert _titles(apply_movie_filters(MOVIES, min_plays=2)) == ["often"]


def test_plays_range():
    assert _titles(apply_movie_filters(MOVIES, min_plays=1, max_plays=1)) == ["once"]


def test_missing_watch_count_treated_as_zero():
    movie = {"title": "legacy", "watch_status": "unwatched", "user": "alice"}
    assert _titles(apply_movie_filters([movie], max_plays=0)) == ["legacy"]
