# Releasing tart2ms

A tart2ms release is self-contained: a local pre-release gate, a version bump
with changelog stamping, then a tag push whose CI publishes to PyPI — with the
task waiting for the workflow, PyPI itself, and the GitHub release.

```console
$ uv run invoke status                       # anything that would make it unsafe
$ uv run invoke version                      # what it would bump to and tag
$ uv run invoke test                         # the gate only
$ uv run invoke release                      # test -> bump -> tag -> monitor
$ uv run invoke release --no-bump            # tag the current version as-is
$ uv run invoke release --version 0.9.5      # tag exactly this (implies --no-bump)
$ uv run invoke release --skip-tests         # emergency: tag without the gate
```

```
 1. gate   flake8 fatal scan (scoped to tart2ms/) + style count
           → smoke conversions of BOTH fixture formats into real MSes
           → pytest tart2ms/tests/ in the project venv
 2. bump   pyproject.toml + CITATION.cff + a CHANGES.md section, committed
           as `release: bump to X.Y.Z`
 3. tag    annotated vX.Y.Z, `git push origin <branch> vX.Y.Z` →
           .github/workflows/deploy_module.yaml builds and publishes to PyPI
 4. watch  wait for the deploy_module *and* testing.yaml runs of the tag to
           go green (failed runs print `gh run view … --log-failed`), poll
           PyPI until the version is served, then create the GitHub release
           from the CHANGES.md section
```

**What this deliberately does *not* do:** ship the release inside the TART
tools image. The image installs `tart2ms>=…` from PyPI, so getting it to users
is a separate, explicit step — rebuild with
`invoke -r ~/tart-telescope/tools release` (loose `>=` pin, so no Dockerfile
edit) and then release tart_cargo with that tag. That rebuild is normally
driven by disko's chain instead; see
[tools/RELEASE.md](https://github.com/tart-telescope/tools/blob/main/RELEASE.md),
[tart_cargo/RELEASE.md](https://github.com/tart-telescope/tart_cargo/blob/main/RELEASE.md)
and [disko/RELEASE.md](https://github.com/tmolteno/disko/blob/master/RELEASE.md).

## Prerequisites

| Need | Why |
|---|---|
| `uv run invoke …` with the `dev` group synced | it carries `invoke`, `plumbum`, `pytest` and `flake8` — the gate runs all four from the project venv |
| `gh`, authenticated | the workflow polls and the GitHub release |
| push rights, and an **up-to-date branch** | see [Version reconciliation](#version-reconciliation); also `git fetch --tags` so the stale-tag guard cannot trip |
| network | the CI/PyPI monitoring polls, and `uv build`-style isolated builds |
| `PYPI_TART2MS_TOKEN` secret | the publish step in `deploy_module.yaml` (already configured) |

> **There is no confirmation prompt.** `invoke release` tags and pushes as soon
> as you run it — run `invoke status` and `invoke version` first.

## The tasks

### `status` — the safety screen

Prints pyproject + CITATION versions, the highest `v*` tag, and what a patch
release would produce, then warns about every state that has bitten before:

- pyproject behind/ahead of the tags (and what that means for the bump);
- a highest tag that is **not reachable from this branch** (the branch
  predates a release made elsewhere);
- the branch being **behind its upstream** (or ahead, i.e. unpushed work);
- modified tracked files *and* untracked files — `release` refuses on any
  `git status --porcelain` line, so a stray `handover.md` would abort it.

### `version`

The one-line projection: `pyproject 0.9.4 -> would bump to 0.9.5 and tag
v0.9.5`, plus how `--no-bump` / `--version X.Y.Z` differ. Uses the same
reconciled base as `release`.

### `test` — the pre-release gate

1. **flake8 fatal scan** — `--select=E9,F63,F7,F82` on `tart2ms` (0 findings
   expected), then the informational style count (~127, never fails the run);
2. **smoke conversions** — `test_data/data_*.json` and `test_data/vis_*.hdf`
   each converted to a real measurement set in a temp dir, asserted to contain
   `table.dat` + the `ANTENNA` subtable;
3. **pytest** — `python -m pytest tart2ms/tests/ -q` via `sys.executable`, so
   it is always the project venv's interpreter (the same suite CI runs).

`release` runs this unless `--skip-tests` (emergency re-tags only — the
docstring says so, and the task prints a warning when skipped).

### `release` — the chain

Flags: `--no-bump`, `--version X.Y.Z` (implies `--no-bump`), `--skip-tests`.

Guards, in the order they fire:

1. `status` preflight (prints the safety report — warnings only);
2. the gate (unless skipped);
3. bump: base is the **reconciled** version (below), and if the resulting tag
   already exists on origin the run aborts ("your tag fetch is stale —
   `git fetch --tags`") rather than re-taking a released tag;
4. **dirty tree** → `RuntimeError` naming the files (untracked included);
5. an existing tag on origin → verified and skipped (idempotent), not re-pushed.

Then it pushes branch + tag, watches the tag's `deploy_module.yaml` **and**
`testing.yaml` runs until green (a failure raises with the
`gh run view <id> -R tart-telescope/tart2ms --log-failed` hint), polls
`https://pypi.org/pypi/tart2ms/<version>/json` until the release is served,
and finally creates the GitHub release from the `## vX.Y.Z` section of
`CHANGES.md` (via a temporary `.release-notes-<version>.md`).

## Version reconciliation

Versions live in `pyproject.toml` + `CITATION.cff`, but the tag namespace is
what publishes, so the next version is the bump of **whichever is higher —
pyproject or the highest `v*` tag anywhere in the repository**:

- `git describe` only sees tags *reachable from the current branch*. `v0.9.4`
  is on `master` while `casacure-backend` predated it, so a pyproject-only
  bump would have tried to re-take a tag that has been on PyPI since
  2026-10-01. Taking the maximum lands on `0.9.5`.
- Prerelease tags (`v0.7.0rc1`, `v0.2.1b3`) parse to their numeric core, so
  they can never break the comparison or the bump.
- Tags are `vMAJOR.MINOR.PATCH` and **are** the publish trigger; never push
  one by hand unless you intend to publish that exact commit.

## CHANGES.md stamping

Sections are newest-first: `## v0.9.4 — Archive search, …` (an em-dash title,
no dates). On a bump, `_stamp_changelog`:

- leaves the file untouched if `## vX.Y.Z` already exists (idempotent —
  hand-written notes win);
- otherwise converts `## Unreleased` (optionally `## Unreleased — <title>`)
  into the new section, leaving a fresh empty `## Unreleased` on top;
- otherwise inserts `## vX.Y.Z — Release vX.Y.Z` above the first section.

If the version being tagged has no section at all, `release` warns — the tag
proceeds, but no GitHub release notes are created.

## CI

| Workflow | Trigger | What it does |
|---|---|---|
| `deploy_module.yaml` | push of a `v*.*.*` tag | `python -m build` → `pypa/gh-action-pypi-publish` with `PYPI_TART2MS_TOKEN` |
| `testing.yaml` | push, pull request, weekly cron | flake8 (fatal + style) and pytest on **3.11/3.12/3.13** |

`testing.yaml`'s matrix was `3.8/3.9/3.10/3.11` while `requires-python` is
`>=3.11,<3.14`, so three of four legs could not even `pip install .`; it now
builds the interpreters the package supports, and uses
`actions/checkout@v7`. `release` watches the tag's run of both workflows.

## Safety

| Guard | Behaviour |
|---|---|
| no prompt | runs immediately — check `status`/`version` first |
| dirty tree | refuses on **any** porcelain line, untracked included |
| re-taken tag | aborts if the computed tag exists on origin (stale fetch) |
| idempotency | a tag already on origin is verified and skipped, not re-pushed |
| `--skip-tests` | prints that the gate is skipped; emergency use only |
| monitoring | publish + test runs must go green; PyPI must serve the version |

## Manual fallback

```console
$ uv run flake8 tart2ms --count --select=E9,F63,F7,F82
$ uv run pytest tart2ms/tests/ -q
$ uv run invoke version                   # what a bump would produce
# edit pyproject.toml version + CITATION.cff + CHANGES.md section
$ git commit -m "release: bump to X.Y.Z" -- pyproject.toml CITATION.cff CHANGES.md
$ git tag -a vX.Y.Z -m vX.Y.Z && git push origin <branch> vX.Y.Z
$ gh run list --workflow deploy_module.yaml --watch
# then create the GitHub release by hand from the CHANGES.md section
```

## Known quirks

* **`uv run pytest` used to run the wrong pytest.** With no dev group, pytest
  was not in the project env, so `uv run` fell back to whatever was on `PATH` —
  a global pytest on another interpreter that could not import the project
  (`ModuleNotFoundError`, seen as "7 errors during collection"). The dev
  group fixes it; the tasks use `sys.executable -m pytest` so they cannot
  regress. The `Makefile`'s `test:` target (`python3 -m pytest`) is still the
  global-interpreter flavour — prefer `uv run invoke test`.
* **`flake8 .` walks the gitignored `venv/` and `.venv/` trees** in a working
  checkout (101 270 findings, all site-packages). The gate is scoped to
  `tart2ms` (0 fatal / ~127 style); CI's clean checkout has no venvs.
* **Fresh measurement sets have no `table.info`.** casacure does not create
  the file python-casacore always wrote — see
  [tmolteno/casacure#23](https://github.com/tmolteno/casacure/issues/23).
  Pre-switch fixtures have one, new conversions do not, which is why the smoke
  test checks `table.dat` + `ANTENNA` instead.
* **Untracked files block `release`** (`git status --porcelain` includes
  `??`); `invoke status` names them so the abort is not a surprise.
* **Release from an up-to-date branch**, and `git fetch --tags` first: a
  branch that trails a release, or a stale tag fetch, is how versions get
  re-taken (both are detected and reported).

## Timing

* gate: seconds once interpreters are cached (~50 unit tests, two conversions,
  two flake8 passes); the first run on a new Python downloads it;
* monitoring: the publish run registers within ~20 s polls (5 min limit),
  each watch polls every 30 s (30 min limit), PyPI up to 15 min;
* the tools-image rebuild that *ships* the release is the slow part of the
  wider chain (up to ~60 min for the arm64 leg) — separate step, see above.

## Troubleshooting

| Message / symptom | Cause / fix |
|---|---|
| `tree is dirty; commit the release … first` | something is modified *or untracked* — check `invoke status` |
| `vX.Y.Z already exists on origin — your tag fetch is stale` | `git fetch --tags`, then re-check with `invoke version` |
| `no deploy_module.yaml run registered for … in 300 s` | the tag push did not reach GitHub (push rules/network) — check <https://github.com/tart-telescope/tart2ms/actions> |
| `<workflow> did not succeed (…)` | read the tail it prints, or `gh run view <id> -R tart-telescope/tart2ms --log-failed` |
| `tart2ms X not on PyPI after 900 s` | the publish ran but the upload failed (token/environment) — inspect the publish step log |
| `warning: CHANGES.md has no '## vX.Y.Z' section` | the tag ships without release notes; add the section (or let the bump stamp it) before tagging |
| `No module named pytest` / `flake8` | `uv sync --dev` (they live in the dev group) |
