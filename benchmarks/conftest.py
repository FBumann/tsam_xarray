import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--large",
        action="store_true",
        default=False,
        help="run large-scale benchmarks (production-sized data, minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "large: large-scale benchmark, runs only with --large"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--large"):
        return
    skip = pytest.mark.skip(reason="large benchmark: pass --large to run")
    for item in items:
        if "large" in item.keywords:
            item.add_marker(skip)
