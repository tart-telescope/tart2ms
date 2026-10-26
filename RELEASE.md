# Releasing tart2ms

A tart2ms release is self-contained: exhaustive tests, a version bump, and a
tag push that makes CI publish to PyPI.

```console
$ uv run invoke release --dry-run     # print the plan, change nothing
$ uv run invoke release               # prompts once, then runs everything
$ uv run invoke release --level minor --yes
```

```
1. test     uv sync --dev + uv lock --check → flake8 fatal gate → pytest on
            3.11, 3.12 and 3.13 → smoke conversions of both fixture formats
            into real measurement sets → uv build
2. bump     the next version in pyproject.toml **and** CITATION.cff, plus a
            CHANGES.md section
3. publish  commit + annotated tag v<version> + push →
            .github/workflows/deploy_module.yaml builds with `python -m
            build` and uploads via pypa/gh-action-pypi-publish
            (PYPI_TART2MS_TOKEN) → wait for that run to go green
```

**What this chain deliberately does *not* do:** ship the release inside the
TART tools image. The image installs `tart2ms>=…` from PyPI, so getting a new
release to users is a separate, explicit step — rebuild the image with
`invoke -r ~/tart-telescope/tools release` (its pin is a loose `>=`, so no
Dockerfile edit is needed) and then release tart_cargo with that tag. That
rebuild is normally driven by disko's release chain instead; see
[tools/RELEASE.md](https://github.com/tart-telescope/tools/blob/main/RELEASE.md),
[tart_cargo/RELEASE.md](https://github.com/tart-telescope/tart_cargo/blob/main/RELEASE.md)
and [disko/RELEASE.md](https://github.com/tmolteno/disko/blob/master/RELEASE.md).

## Prerequisites

| Need | Why |
|---|---|
| `uv`, and `uv run invoke …` | the `dev` dependency group carries `pytest`, `flake8`, `invoke` and `plumbum`, so the project venv always has them |
| `gh` CLI, authenticated | `publish` waits for the deploy_module run; otherwise pass `--no-wait` |
| push rights to `tart-telescope/tart2ms` | the release tags and pushes |
| `PYPI_TART2MS_TOKEN` secret | the publish step in `deploy_module.yaml` (already configured) |
| an up-to-date branch | see [Version reconciliation](#version-reconciliation) — releasing from a branch that trails the release tag is how versions get re-taken |

## The tasks

```console
$ uv run invoke --list
  bump      Set the next version in pyproject + CITATION and open a CHANGES
            section.
  publish   Commit the bump, tag vX.Y.Z and push: CI publishes to PyPI.
  release   Full release: exhaustive tests -> version bump -> CI publish.
  status    Show version, tags, and anything that would make a release unsafe.
  test      Exhaustive local tests — everything a release must pass first.
```

### `test` — the gate

`uv sync --dev`, `uv lock --check`, the flake8 **fatal** gate
(`--select=E9,F63,F7,F82`) plus the informational style count (127 findings;
never fails the run), `pytest` on **each** of 3.11/3.12/3.13 (matching
`requires-python = ">=3.11,<3.14"` and `.github/workflows/testing.yaml`), then
real conversions into measurement sets:

```text
tart2ms --help
tart2ms --json test_data/data_2019_08_04_21_38_31_UTC.json --ms <tmp>/smoke_json.ms
tart2ms --hdf  test_data/vis_2026-06-12_04_25_46.149086.hdf --ms <tmp>/smoke_hdf.ms \
        --rephase obs-midpoint --single-field
```

each asserted to have produced an MS, and finally `uv build` (the distributions
deploy_module will rebuild). `--quick` drops to one interpreter and skips the
build.

> The smoke check looks for `table.dat` + the `ANTENNA` subtable rather than
> `table.info`: **casacure does not create `table.info`**, while python-casacore
> did — the committed fixtures have one, a freshly converted MS does not. See
> [Known quirks](#known-quirks).

### `bump`

`--level patch|minor|major` (default `patch`) sets the new version in
`pyproject.toml` (keeping its single-quote style) and in `CITATION.cff`, then
ensures `CHANGES.md` has a `## v<version> — …` section for it: an existing
section is kept (entries are written while work lands), otherwise one is
drafted from `git log <tag>..HEAD` and flagged for review. `--dry-run` previews,
including the commits that would land in the section.

### `publish`

Refuses if the tag already exists, or if tracked files other than
`pyproject.toml`, `CHANGES.md` and `CITATION.cff` are modified (untracked
scratch — stray `.ms` outputs, caches — is only reported). Then commits
`Release <version>`, creates the annotated tag, pushes with `--follow-tags`,
and waits (default) for the `deploy_module.yaml` run on that tag via
`gh run list`. `--no-wait` skips the wait, `--yes` skips the prompt.

### `status`

Everything that would make a release unsafe, in one screen: the versions
(pyproject, CITATION, highest tag), whether pyproject is behind or ahead of
the tags, whether the highest tag is reachable from the current branch, and
how far the branch is from its upstream.

## Version reconciliation

Versions live in two files and one tag namespace, and they have drifted
before, so the release logic reconciles rather than trusts:

* **The version is `pyproject.toml`'s `[project].version`**, kept in step with
  `CITATION.cff`. The old `[tool.hatch.version] path = "tart2ms/__init__.py"`
  pointer was removed with this change: that file has never carried
  `__version__`, and hatchling was using the static field anyway (verified —
  the wheel still builds and is still `tart2ms-<version>`).
* **The next version is the bump of whichever is higher, the tag or the
  file.** `git describe` only sees tags reachable from the current branch, and
  releases cut elsewhere are invisible to it: `v0.9.4` lives on `master` while
  `casacure-backend` predates it, so a describe-based patch bump would have
  tried to re-take `v0.9.4` (which exists, on PyPI since 2026-10-01). Taking
  the maximum lands on `0.9.5`.
* **`status` and `bump` say so out loud** when pyproject trails the tag, when
  the highest tag is not reachable from this branch, or when the branch is
  behind its upstream — release from an up-to-date branch.
* **Tags are `vMAJOR.MINOR.PATCH` and are the publish trigger.** Never push one
  by hand unless you intend to publish that exact commit.
* `CHANGES.md` uses one section per release — `## v0.9.3 — Performance and
  dask scheduler` — with no dates.

## CI

| Workflow | Trigger | What it does |
|---|---|---|
| `deploy_module.yaml` | push of a `v*.*.*` tag | `python -m build` → `pypa/gh-action-pypi-publish` with `PYPI_TART2MS_TOKEN` |
| `testing.yaml` | push, pull request, weekly cron | flake8 (fatal gate + style) and pytest on 3.11/3.12/3.13 |

`testing.yaml`'s matrix was 3.8–3.11 while `requires-python` is `>=3.11,<3.14`,
so three of its four legs could not even `pip install .`; it now builds the
three interpreters the package supports (and uses `actions/checkout@v7`).

## Safety

| Guard | Behaviour |
|---|---|
| prompt | `release` asks once; `publish` asks when run alone; `--yes` bypasses |
| `--dry-run` | `release` and `bump` print the full plan and exit |
| clean tree | only `pyproject.toml`, `CHANGES.md` and `CITATION.cff` may be modified; untracked scratch is reported, never blocking |
| no re-tag | publishing a tag that already exists is refused |
| wait | `publish` polls the workflow run on the tag (default; `--no-wait` to skip) |

## Manual fallback

```console
$ uv run pytest && uv run flake8 tart2ms --select=E9,F63,F7,F82 --count
# edit pyproject.toml version and CITATION.cff; add a CHANGES.md section
$ git commit -am "Release X.Y.Z"
$ git tag -a vX.Y.Z -m "Release X.Y.Z" && git push --follow-tags origin <branch>
$ gh run list --workflow deploy_module.yaml --watch
# then rebuild the tools image and release tart_cargo — see the links above
```

## Known quirks

* **`uv run pytest` used to run the wrong pytest.** With no dev group,
  pytest was not in the project env, so `uv run` fell back to whatever was on
  `PATH` — a global pytest on another interpreter that could not import the
  project (`ModuleNotFoundError: tart2ms`) or its dependencies. The `dev`
  group fixes that; never rely on PATH tools for the gate.
* **`flake8 .` walks the gitignored `venv/` and `.venv/` trees** in a working
  checkout (101270 findings, all site-packages). The gate is therefore scoped
  to `tart2ms` (0 fatal / 127 style). CI's `flake8 .` is unaffected — a clean
  checkout has no venvs.
* **Fresh measurement sets have no `table.info`.** casacure, the Rust casacore
  replacement this branch writes through, does not create the file that
  python-casacore always wrote; fixtures converted before the switch have one,
  new conversions do not. Tools that expect `table.info` (CASA-side readers)
  may notice — raised upstream with casacure, and why the smoke test checks
  `table.dat` + `ANTENNA` instead.
* **`uv build` needs the network** (it fetches `hatchling` into an isolated
  build env). The conversion smoke tests are fully offline.

## Timing

* `test`: roughly a minute once the interpreters are cached — 36 unit tests
  per interpreter plus two conversions and a build; the first run on a new
  Python downloads it.
* `publish` wait: up to 30 minutes, usually about a minute.
* The tools image rebuild that ships the release (separate step) is the slow
  part of the wider chain: up to ~60 minutes for the arm64 leg.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `publish workflow failed: <url>` | the tag is already pushed — **do not re-tag**. Fix on a branch, bump again, release the next version. |
| `tag v… already exists` at publish | the computed version was taken; `invoke status` shows the reconciliation — bump again |
| `pyproject is BEHIND v…` / `tag is NOT reachable` | you are on a branch that trails the release; update from upstream before releasing |
| `gh CLI is required to wait…` | install `gh`, or pass `--no-wait` and watch CI yourself |
| `working tree has unrelated changes` | commit or stash; untracked `.ms` outputs are only reported |
| `error sending request … Temporary failure in name resolution` during `uv build` | no network; retry |
| prompt does not appear | `release` reads from the terminal; use `--yes` in non-interactive runs |
