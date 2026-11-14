"""--fail-on-skip: a skipped check must fail a release run instead of passing quietly."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SUITES = {
    "test-skip": {"test_s.py": "import pytest\n\ndef test_tool():\n    pytest.skip('reference tool missing')\n"},
    "marker-skip": {"test_s.py": "import pytest\n\n@pytest.mark.skipif(True, reason='no tool')\ndef test_tool():\n    pass\n"},
    "module-skip": {"test_s.py": "import pytest\npytest.importorskip('module_that_is_not_installed')\n"},
    "nested-conftest-skip": {"sub/conftest.py": "import pytest\npytest.importorskip('module_that_is_not_installed')\n",
                             "sub/test_s.py": "def test_never_runs():\n    pass\n"},
    "teardown-skip": {"test_s.py": "import pytest\n\n@pytest.fixture\ndef tool():\n    yield\n    pytest.skip('cleanup tool missing')\n\n"
                                    "def test_tool(tool):\n    pass\n"},
    "two-skips": {"test_s.py": "import pytest\n\ndef test_a():\n    pytest.skip('a')\n\ndef test_b():\n    pytest.skip('b')\n"},
    "skip-then-exit-3": {"test_s.py": "import pytest\n\ndef test_a():\n    pytest.skip('a')\n\n"
                                       "def test_b():\n    pytest.exit('stop', returncode=3)\n"},
    "xfail": {"test_s.py": "import pytest\n\n@pytest.mark.xfail(reason='known', strict=True)\ndef test_known():\n    assert False\n"},
    "no-skip": {"test_s.py": "def test_ok():\n    pass\n"},
}


def run_suite(tmp_path: Path, name: str, *flags: str) -> subprocess.CompletedProcess[str]:
    shutil.copy2(ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_other.py").write_text("def test_passes():\n    pass\n", encoding="utf-8")
    for relative, text in SUITES[name].items():
        (tmp_path / relative).parent.mkdir(exist_ok=True)
        (tmp_path / relative).write_text(text, encoding="utf-8")
    # The caller's PYTEST_ADDOPTS (for example --fail-on-skip) must not reach the inner runs.
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *flags, str(tmp_path)],
                          cwd=tmp_path, text=True, capture_output=True, timeout=60, env=env)


@pytest.mark.parametrize("name", ["test-skip", "marker-skip", "teardown-skip", "module-skip", "nested-conftest-skip"])
def test_skips_in_each_phase_and_scope_fail_with_the_flag(tmp_path, name):
    assert run_suite(tmp_path, name).returncode == 0
    result = run_suite(tmp_path, name, "--fail-on-skip")
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["xfail", "no-skip"])
def test_xfail_and_clean_runs_still_pass(tmp_path, name):
    result = run_suite(tmp_path, name, "--fail-on-skip")
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_skipped_check_is_named(tmp_path):
    result = run_suite(tmp_path, "two-skips", "--fail-on-skip")
    assert "--fail-on-skip: skipped test_s.py::test_a, test_s.py::test_b" in result.stdout


def test_an_existing_failure_status_is_kept(tmp_path):
    assert run_suite(tmp_path, "skip-then-exit-3", "--fail-on-skip").returncode == 3
