"""Smoke test checking the package version string."""

from __future__ import annotations

from harmonizepy import __version__


def test_version_is_defined() -> None:
    assert __version__ == "0.2.0"
