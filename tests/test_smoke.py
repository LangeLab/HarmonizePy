"""Smoke test checking the package version string."""

from __future__ import annotations

from harmonizepy import __version__


def test_version_is_defined() -> None:
    """Package version string must be ``0.2.0``.

    Failure condition: the version was bumped or a dev install does
    not match the released version.
    """
    assert __version__ == "0.2.0"
