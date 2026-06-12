"""T0.1 scaffold smoke tests: the package imports and the marker set works."""

import pytest

import acagent

pytestmark = [pytest.mark.unit, pytest.mark.phase0]


def test_package_imports() -> None:
    assert acagent.__version__


def test_session_subpackage_imports() -> None:
    import acagent.session  # noqa: F401
