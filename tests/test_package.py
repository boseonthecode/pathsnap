"""Package import smoke test for the scaffolding step."""

import pathsnap


def test_package_imports_and_exposes_version() -> None:
    assert pathsnap.__version__
