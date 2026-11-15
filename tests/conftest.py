from __future__ import annotations

import json
from pathlib import Path

import pytest

from quietcycle import PipelineRequest

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--fail-on-skip", action="store_true", default=False,
                     help="Fail the run if any test, module or directory was skipped.")


class _FailOnSkip:
    """Registered as a plugin, not as conftest hooks, so it sees reports from every directory."""

    def __init__(self) -> None:
        self.skipped: list[str] = []

    def _count(self, report: pytest.CollectReport | pytest.TestReport) -> None:
        # xfail is reported as skipped with a wasxfail marker; it is a separate outcome.
        if report.skipped and not hasattr(report, "wasxfail"):
            self.skipped.append(report.nodeid)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        self._count(report)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._count(report)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.skipped and session.exitstatus == pytest.ExitCode.OK:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                reporter.write_line(f"--fail-on-skip: skipped {', '.join(self.skipped)}", red=True)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--fail-on-skip"):
        config.pluginmanager.register(_FailOnSkip(), "fail-on-skip")


@pytest.fixture
def request_data():
    return json.loads((ROOT / "examples/request.json").read_text())


@pytest.fixture
def sample(request_data):
    return PipelineRequest.model_validate(request_data)
