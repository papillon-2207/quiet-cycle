"""--fail-on-skip: a skipped check must fail a release run instead of passing quietly."""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tomllib
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


def test_repository_check_does_not_depend_on_the_locale_encoding(tmp_path):
    # Windows defaults to cp1252, which cannot decode the UTF-8 READMEs; an ASCII locale reproduces
    # that on Linux. Á is UTF-8 c3 81, and cp1252 has no 81, so the text fails under both. The check
    # runs on a minimal copy, so the test also works from the sdist.
    for name in ("scripts", "src", "tests", "examples", "docs"):
        (tmp_path / name).mkdir()
    shutil.copy2(ROOT / "scripts" / "check_repository.py", tmp_path / "scripts")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("# Quiet Cycle（測試，Á）\n" + pyproject, encoding="utf-8")
    (tmp_path / "README.md").write_text("# 測試 Á\n\n[專案設定](pyproject.toml)\n", encoding="utf-8")
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0"}
    result = subprocess.run([sys.executable, str(tmp_path / "scripts" / "check_repository.py")],
                            cwd=tmp_path, text=True, capture_output=True, timeout=60, env=env)
    assert result.returncode == 0, result.stderr[-500:]
@pytest.mark.parametrize("script", ["check_artifacts.py", "check_repository.py"])
def test_release_scripts_refuse_to_run_without_assertions(tmp_path, script):
    # Their checks are assert statements, which -O and PYTHONOPTIMIZE remove; they then reported passes.
    env = {**os.environ, "PYTHONOPTIMIZE": "1"}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                            cwd=tmp_path, text=True, capture_output=True, timeout=60, env=env)
    assert result.returncode != 0
    assert f"{script}: assertions are disabled" in result.stderr
    assert result.stdout == ""


def test_scripts_name_the_encoding_of_every_text_file_they_open():
    # Without it a script reads and writes in the locale's encoding, such as cp1252 on Windows.
    unnamed = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            receiver = getattr(getattr(node.func, "value", None), "id", None)
            if name not in ("open", "read_text", "write_text") or receiver == "tarfile":
                continue
            mode = next((k.value for k in node.keywords if k.arg == "mode"), None)
            if mode is None and name == "open":
                position = 1 if isinstance(node.func, ast.Name) else 0  # open(path, mode), Path.open(mode)
                mode = node.args[position] if len(node.args) > position else None
            binary = isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value
            if binary or any(k.arg == "encoding" for k in node.keywords):
                continue
            unnamed.append(f"{path.name}:{node.lineno}")
    assert unnamed == []


def test_lowest_constraints_pin_the_lower_bound_of_every_declared_dependency():
    # CI installs constraints-lowest.txt on Python 3.11. Without this test, the lower bound of a
    # requirement added to pyproject.toml, or a bound lowered there, would go untested in that job.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = [*project["dependencies"], *(r for extra in project["optional-dependencies"].values() for r in extra)]
    bounds = [re.fullmatch(r"([A-Za-z0-9._-]+)>=([0-9.]+),<[0-9.]+", r) for r in declared]
    assert all(bounds), declared
    lines = (ROOT / "constraints-lowest.txt").read_text(encoding="utf-8").splitlines()
    pins = sorted(tuple(line.split("==")) for line in lines if line and not line.startswith("#"))
    assert pins == sorted({m.groups() for m in bounds if m})
