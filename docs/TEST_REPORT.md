# Test report

This report covers commit `a570802`, which sets the version to 0.4.1; the changes since 0.4.0 are
listed in [CHANGELOG.md](../CHANGELOG.md). The checks ran on 2026-11-22 with synthetic
fixtures only; no clinical accuracy, contraceptive effectiveness or patient safety was evaluated.

The 0.4.0 report, for commit `88eb038`, is in commit `2e01dd5`. The 0.3.0 report is kept in
[archive/TEST_REPORT-0.3.0.md](archive/TEST_REPORT-0.3.0.md), and the evidence it cites is in commit
`06d3f80`. The test, coverage, contract, repository, mutation, build, installed-wheel and
environment files in `evidence/` now come from the Linux Python 3.13 run below. The benchmark files
are unchanged since 0.3.0, `evidence/actions-pins.json` was checked by hand, and
`evidence/matrix.json` summarizes every local run and the CI run.

## Linux, Python 3.11 to 3.14

Each version used a fresh copy of the commit and a new virtual environment created with uv.
`pip install -c constraints-contracts.txt -e ".[dev,api,reference]"` ran with the pip cache
disabled, so every distribution that pip installed was downloaded from PyPI. The interpreters were
CPython 3.11.16, 3.12.14, 3.13.16 and 3.14.7 on x86-64, and every check below gave the same result
on all four. `evidence/matrix.json` lists the versions of the main packages in each run; Python 3.11
resolved NumPy 2.4.6, the others 2.5.5.

| Check | Result | Evidence |
| --- | --- | --- |
| Full pytest with `--fail-on-skip` | 377 passed, none skipped or failed; 1 warning | `evidence/pytest.log`, `pytest.xml` |
| Combined statement/branch coverage | 89.78%; 2,401 statements and 886 branches | `evidence/coverage.json` |
| Contract drift | 11 JSON Schemas and 1 default OpenAPI document unchanged | `evidence/contracts.log` |
| Syntax / exports / documentation links | 60 Python files parse for 3.11; 29 core and 18 learning exports | `evidence/repository-check.json` |
| Strict mypy | No issues in 31 source files | `evidence/matrix.json` |
| Ruff | All checks passed | `evidence/matrix.json` |
| Selected regression mutations | 114/114 caught | `evidence/mutations.json` |
| Synthetic learning benchmark | Output byte-identical to the committed evidence | `evidence/learning-benchmark.json` |
| `python -m build` | Wheel and sdist; setuptools 84.2.1 in an isolated environment | `evidence/build.log` |
| `twine check --strict` | Both files passed | `evidence/matrix.json` |
| Installed wheel, outside checkout | 24 checks, none skipped, including Node and live learned HTTP | `evidence/artifact-smoke.json` |

`evidence/environment.json` lists the package versions of the 3.13 run. Every check exited with
status 0. Only the build wrote to stderr: its progress messages and setuptools warnings about
excluded paths that do not exist. The pytest warning is a `DeprecationWarning` raised inside
Starlette's test client (`anyio.abc.BlockingPortal`), not in this package.

## GitHub Actions

[Run 44995609311](https://github.com/papillon-2207/quiet-cycle/actions/runs/44995609311) ran on the
push of the same commit. Each job ran the install, pytest with `--fail-on-skip`, contract drift, the
repository check, strict mypy, Ruff, the build, the installed-wheel check, `twine check --strict`
and the benchmark with a check that `evidence/` did not change; the Linux jobs also ran the
mutations (114 of 114 caught). All of these steps succeeded. Pytest reported the same warning and
the same coverage total as the local runs.

| Job | Runner image | Python | pytest | Installed wheel |
| --- | --- | --- | --- | --- |
| ubuntu-latest, 3.11 | ubuntu-26.04 | 3.11.18 | 377 passed | 24 passed, none skipped |
| ubuntu-latest, 3.12 | ubuntu-26.04 | 3.12.16 | 377 passed | 24 passed, none skipped |
| ubuntu-latest, 3.13 | ubuntu-26.04 | 3.13.18 | 377 passed | 24 passed, none skipped |
| ubuntu-latest, 3.14 | ubuntu-26.04 | 3.14.10 | 377 passed | 24 passed, none skipped |
| macos-latest, 3.13 | macos-26-arm64 | 3.13.17 | 377 passed | 24 passed, none skipped |
| windows-latest, 3.13 | windows-2025-vs2026 | 3.13.17 | 377 passed | 24 passed, none skipped |

The workflow pins actions/checkout and actions/setup-python to full commits; on 2026-11-22 these
were still the commits of tags v7 (v7.0.1) and v7 (v7.1.0), as `evidence/actions-pins.json` records.
Both run on Node.js 24, and no job had a warning annotation.

## Inside the suite

| Check | Size | Location |
| --- | --- | --- |
| Legacy JS/Python parity | 300 generated requests | `tests/test_reference.py` |
| Integer kernel / display allocation | 1,400 generated cases | `tests/test_reference.py` |
| Independent duration countdown oracle | 200 exact models | `tests/test_research.py` |
| Ridge normal-equation check | 60 generated systems over 3 positive penalties | `tests/test_learning.py` |
| Independent numerical ridge comparison | One NumPy linear solve, error below 1e-12 | `tests/test_learning.py` |
| Conformal finite-sample enumeration | All 120 orderings; expected 4/5 coverage | `tests/test_learning.py` |
| Different Python hash seeds | Same serialized prediction from one fitted model for 0, 7 and 123456 | `tests/test_learning.py` |

The generated cases are loops within tests, not extra test functions or human subjects. The
normal-equation certificate checks the selected ridge objective, not whether it is the right model
for human cycles. Coverage is not proof of no defects.

The suite has grown from 335 to 377 tests since 0.4.0: 12 in a new file for names over the
96-character limit, and 30 in existing ones for the CLI import options, adapter error codes, labels
read in each event's own offset, outcomes known at the landmark, and the release scripts' encodings
and refusal to run without assertions. The learning-1.0 digests pinned in `tests/test_learning.py`
are unchanged.

## Three fixed synthetic scenarios

Each uses 36 training, 18 tuning, 24 calibration and 24 test subjects/cases, with
separated calendar blocks and no rejected fixture cases. The generator specifies
which measurements correlate with the target; it is not a simulator validated
against human physiology. The results are unchanged since 0.3.0.

| Scenario | Learned test MAE | Baseline MAE, same cases | Interpretation |
| --- | --- | --- | --- |
| Stable constructed signal | 13/12 days (1.0833) | 9/4 days (2.25) | The full pipeline can learn the supplied relation |
| Unrelated target component | 77/24 days (3.2083) | 11/4 days (2.75) | Tuning selection can overfit; it is not guaranteed better on test |
| Relation reverses only in test | 19/4 days (4.75) | 8/3 days (2.6667) | Drift can defeat the learned association and interval coverage |

All selected `wearable-ridge-1` in this fixture. This is not a conclusion that
wearables are universally superior. `evidence/learning-benchmark.json` retains the
full attempt counts, interval widths, coverage, tails, selected model and hashes.
There was no retuning to remove the adverse rows. These are software experiments,
not an empirical superiority claim or a clinical sample-size study.

## Boundary of installation evidence

The installed-wheel check installs the wheel with `pip --no-deps --no-index --target` outside the
source tree. This isolates project imports but reuses the environment's dependencies. The build
used `python -m build`, which installs setuptools into an isolated environment. The builds are not
byte-reproducible: local builds of the same commit gave different wheel hashes.

## Not executed / not established

- Windows and macOS ran only in GitHub Actions, and only with Python 3.13. The mutations ran only
  on Linux; the workflow skips them on Windows and macOS.
- Every run used the pins in `constraints-contracts.txt` (Pydantic 2.13.4, FastAPI 0.128.2,
  HTTPX 0.28.1). The lowest versions the package declares, such as Pydantic 2.12 or NumPy 1.26,
  were not tested.
- Nothing was uploaded. The package is not on PyPI, and `twine upload` was not run.
- No independently authorized human-data fitting, external validation, prospective
  trial, sensor hardware validation, subgroup parity study, comprehensive FHIR
  certification, regulatory assessment or independent security review occurred.
- There is no pretrained clinical artifact and no unrestricted all-day forecaster.
  Default learned artifacts operate at 20 elapsed days; other landmarks need their
  own training and evaluation.

## Reproduce

```sh
python -m pip install -c constraints-contracts.txt -e ".[dev,api,reference]"
python -m pytest --fail-on-skip --cov=quietcycle --cov-report=term
python scripts/generate_contracts.py --check
python scripts/check_repository.py
python scripts/benchmark_learning.py
python scripts/run_mutations.py
python -m mypy src/quietcycle
python -m ruff check src tests examples scripts
python -m build
python scripts/check_artifacts.py
python -m twine check --strict dist/*
```

The evidence run added `-q -p no:cacheprovider --cov-report=json:evidence/coverage.json
--junitxml=evidence/pytest.xml` to pytest. `constraints-tested.txt` records the versions observed
in the 3.13 run, not a universal lock file. `MANIFEST.sha256` lists the SHA-256 of every tracked
file in the commit that added this report, except itself.
