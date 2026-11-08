"""Shared test setup.

Offline tests run by default. Tests marked `live` make a handful of real API calls and run only
with `pytest --live`; each one is skipped when the key it needs is missing from .env.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="run tests that make real (small) API calls")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live test: run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def require_key():
    """Skip a live test when its API key is absent, naming the variable to add to .env."""
    def check(name):
        if not os.getenv(name):
            pytest.skip(f"{name} is not set in .env")
        return os.environ[name]
    return check
