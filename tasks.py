"""Release orchestration for tart2ms.

Following the example of ../casacure/tasks.py: `invoke release` runs the
pre-release gate locally before any tag is pushed to GitHub, then lets CI
publish and reports the outcome, so a broken release becomes a local failure
the operator sees first.

Usage (from the tart2ms repo root, with a venv that has invoke + plumbum):

    invoke version                      # show what the release would tag
    invoke status                       # everything that would make it unsafe
    invoke test                         # the pre-release gate only
    invoke release                      # bump patch, stamp CHANGES.md, tag + push
    invoke release --no-bump            # tag the current version as-is
    invoke release --version 0.9.5      # explicit version (implies --no-bump)
    invoke release --skip-tests         # tag without the local gate (emergency)

`invoke release` writes the release commit for you: the patch version in
`pyproject.toml` and `CITATION.cff`, and the `CHANGES.md` heading (the
`## Unreleased` entries move under `## vX.Y.Z - <title>`, with a fresh empty
`## Unreleased` left on top), so the changelog no longer has to be edited by
hand at release time.  The title of the new section is the text after the
first em dash in the Unreleased heading if there is one, else `Release
X.Y.Z`.

The release chain:
  1. pre-release gate: the flake8 fatal-error scan (scoped to `tart2ms/` —
     `flake8 .` walks the gitignored venv trees), smoke conversions of BOTH
     fixture formats into real measurement sets, then
     `python -m pytest tart2ms/tests/` in the current interpreter (the same
     suite CI runs; use `uv run invoke test` to get the project venv).
  2. version bump + CHANGES.md stamp + release commit (`release: bump to
     X.Y.Z`).
  3. annotated tag vX.Y.Z and `git push origin <branch> vX.Y.Z`; the tag
     push triggers `.github/workflows/deploy_module.yaml` which builds and
     publishes the package to PyPI.  (The `testing.yaml` suite runs on the
     push of the release commit as well.)
  4. monitor: wait for every workflow run of the tag to register and go
     green, then poll PyPI until the new version is live, then create the
     GitHub release from the CHANGES.md section.

Version reconciliation: the next version is the bump of **whichever is
higher, pyproject's version or the highest `v*` tag anywhere in the repo**.
`git describe` only sees tags reachable from the current branch, and
releases cut elsewhere are invisible to it — v0.9.4 lives on master while
casacure-backend predated it — so a pyproject-only bump could re-take an
existing tag. Prerelease tags (v0.7.0rc1) parse to their numeric core.
`invoke status` reports this, plus tag reachability and upstream drift.

Idempotent: a tag already on origin is verified and skipped, not re-pushed.
Requires: invoke, plumbum, pytest and flake8 (all in the dev dependency
group), git, gh (authenticated), and network access for the CI/PyPI polls.
See RELEASE.md for the full process.
"""
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from invoke import task
from plumbum import FG, local

REPO = Path(__file__).parent
REPO_SLUG = "tart-telescope/tart2ms"
PUBLISH_WORKFLOW = "deploy_module.yaml"
TEST_WORKFLOW = "testing.yaml"


# ---------------------------------------------------------------------------
# version + changelog helpers
# ---------------------------------------------------------------------------

def _read_version() -> str:
    text = (REPO / "pyproject.toml").read_text()
    m = re.search(r"^version\s*=\s*'([^']+)'", text, re.MULTILINE)
    return m.group(1) if m else "?"


def _bump_patch(version: str) -> str:
    major, minor, patch = version.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def _version_key(version: str) -> tuple[int, int, int]:
    """Numeric core of any version-ish string: '0.9.4' and 'v0.7.0rc1' both
    give (major, minor, patch); anything unparseable sorts first."""
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
    return tuple(int(group) for group in m.groups()) if m else (0, 0, 0)


def _highest_tag() -> str:
    """Highest v* tag anywhere in the repository (not just on this branch).

    `git describe` only sees tags reachable from HEAD, which is exactly how a
    stale branch re-takes a release: v0.9.4 lives on master while
    casacure-backend predated it, so a pyproject-only bump would have tried to
    tag v0.9.4 a second time.
    """
    out = local["git"]["-C", str(REPO), "tag", "-l", "v*"].run(retcode=None)[1]
    tags = out.split()
    return max(tags, key=_version_key) if tags else ""


def _reconciled_version() -> str:
    """Base for a bump: the higher of pyproject's version and the highest tag,
    normalised to x.y.z so a prerelease tag can never break `_bump_patch`."""
    key = max(_version_key(_read_version()), _version_key(_highest_tag()))
    return ".".join(str(part) for part in key)


def _bump_version(new: str) -> None:
    """Rewrite the version in pyproject.toml and CITATION.cff."""
    p = REPO / "pyproject.toml"
    s = p.read_text()
    s = re.sub(r"^version\s*=\s*'[^']+'", f"version = '{new}'", s, count=1, flags=re.MULTILINE)
    p.write_text(s)

    p = REPO / "CITATION.cff"
    s = p.read_text()
    s = re.sub(r"(?m)^version: .*$", f"version: {new}", s, count=1)
    p.write_text(s)


def _changelog_section(version: str) -> str | None:
    """Return the CHANGES.md section text for a version, if any."""
    s = (REPO / "CHANGES.md").read_text()
    m = re.search(rf"^## v{re.escape(version)}\b.*?(?=^## |\Z)", s,
                  re.MULTILINE | re.DOTALL)
    return m.group(0).strip() if m else None


def _stamp_changelog(new: str) -> bool:
    """Move the `## Unreleased` entries under `## vX.Y.Z - <title>`.

    CHANGES.md is newest-first, so a release turns the entries accumulated
    under `## Unreleased` into the released version's own section and leaves
    a fresh, empty `## Unreleased` on top.  Idempotent and respectful of
    hand-written notes: if a `## vX.Y.Z` section already exists the file is
    left untouched and False is returned.
    """
    p = REPO / "CHANGES.md"
    s = p.read_text()
    if re.search(rf"^## v{re.escape(new)}\b", s, re.MULTILINE):
        return False

    unreleased = re.search(r"^## Unreleased[ \t]*(?:—\s*(.*?))?[ \t]*\n",
                           s, re.MULTILINE)
    title = (unreleased.group(1) or "").strip() if unreleased else ""
    heading = f"## v{new}" + (f" — {title}" if title else f" — Release {new}") + "\n"

    if unreleased:
        s = s[: unreleased.end()] + "\n" + heading + s[unreleased.end():]
    else:
        first = re.search(r"^## ", s, re.MULTILINE)
        if not first:
            return False
        s = s[: first.start()] + heading + "\n" + s[first.start():]
    p.write_text(s)
    return True


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return local["git"]["-C", str(REPO), *args]()


def _branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD").strip()


def _tag_exists(tag: str) -> bool:
    out = _git("ls-remote", "--tags", "origin", f"refs/tags/{tag}")
    return bool(out.strip())


def _remote_sha() -> str:
    return _git("ls-remote", "origin", "HEAD").split()[0]


# ---------------------------------------------------------------------------
# CI/CD + PyPI monitoring
# ---------------------------------------------------------------------------

def _gh(*args: str) -> str:
    return local["gh"][*args]()


def _workflow_runs(workflow: str, limit: int = 10) -> list[dict]:
    out = _gh("run", "list", "-R", REPO_SLUG, "--workflow", workflow,
              "--limit", str(limit),
              "--json", "databaseId,headBranch,status,conclusion")
    return json.loads(out)


def _wait_for_ci(tag: str, timeout_s: int = 1800) -> dict[str, int]:
    """Wait for every workflow run of the tag to register, return name->id.

    Runs are matched by headBranch, which for a tag push is the tag itself;
    a bare --limit 1 would see whatever ran last -- often the previous
    release's still-green run.  The testing.yaml suite also runs on the push
    of the release commit (headBranch = branch), so those runs are picked up
    too when their SHA is the tagged commit.
    """
    start = time.time()
    publish_id = None
    test_id = None
    while True:
        for r in _workflow_runs(PUBLISH_WORKFLOW):
            if r["headBranch"] == tag:
                publish_id = r["databaseId"]
                break
        # testing.yaml triggers on every push, so the tag push gets its own
        # run with headBranch == tag. Grab it when it registers — a missing
        # test run only skips that watch, it never gates the publish wait
        # (previously test_id was never assigned, so this run went unwatched).
        for r in _workflow_runs(TEST_WORKFLOW):
            if r["headBranch"] == tag:
                test_id = r["databaseId"]
                break
        if publish_id is not None:
            break
        if time.time() - start > 300:
            raise RuntimeError(
                f"no {PUBLISH_WORKFLOW} run registered for {tag} in 300 s; "
                f"check https://github.com/{REPO_SLUG}/actions")
        print(f"  waiting for the {PUBLISH_WORKFLOW} run of {tag} to register...")
        time.sleep(20)
    return {PUBLISH_WORKFLOW: publish_id, TEST_WORKFLOW: test_id}


def _watch_run(name: str, run_id: int, timeout_s: int = 1800) -> None:
    print(f"  {name} (run {run_id}):")
    start = time.time()
    while True:
        line = _gh("run", "view", str(run_id), "-R", REPO_SLUG,
                   "--json", "status,conclusion",
                   "--jq", '"\\(.status) \\(.conclusion)"').strip()
        status, _, conclusion = line.partition(" ")
        if conclusion == "success":
            print(f"    {name}: {line}")
            return
        if status == "completed":  # failure / cancelled / timed_out / skipped
            raise RuntimeError(
                f"{name} did not succeed ({line}); see "
                f"gh run view {run_id} -R {REPO_SLUG} --log-failed")
        if time.time() - start > timeout_s:
            raise RuntimeError(f"{name} timed out after {timeout_s}s")
        print(f"    {name}: {line}")
        time.sleep(30)


def _wait_for_pypi(version: str, timeout_s: int = 900) -> None:
    """Poll PyPI until the version is live (the index lags the workflow)."""
    url = f"https://pypi.org/pypi/tart2ms/{version}/json"
    start = time.time()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                if resp.status == 200:
                    print(f"  PyPI is serving tart2ms {version}")
                    return
        except Exception:
            pass
        if time.time() - start > timeout_s:
            raise RuntimeError(
                f"tart2ms {version} not on PyPI after {timeout_s}s; check the "
                f"publish run's log for a token/publish failure")
        print(f"  waiting for tart2ms {version} on PyPI...")
        time.sleep(20)


# ---------------------------------------------------------------------------
# pre-release tests
# ---------------------------------------------------------------------------

def _run_pytest() -> None:
    # Use the interpreter running invoke (the project venv), not whatever
    # `python` happens to be on PATH.
    local[sys.executable]["-m", "pytest", "tart2ms/tests/", "-q"] & FG


def _run_flake8() -> None:
    """The fatal-error gate CI runs, scoped to the package.

    `flake8 .` would walk the gitignored venv/ and .venv/ trees in a working
    checkout (101k findings, all site-packages), so the scope is `tart2ms`;
    a clean CI checkout has no venvs, so this matches what CI sees.
    """
    local[sys.executable]["-m", "flake8", "tart2ms", "--count",
                          "--select=E9,F63,F7,F82", "--show-source",
                          "--statistics"] & FG
    local[sys.executable]["-m", "flake8", "tart2ms", "--count", "--exit-zero",
                          "--max-complexity=10", "--max-line-length=127",
                          "--statistics"] & FG


def _run_smoke_conversions() -> None:
    """Convert both fixture formats into real measurement sets (offline).

    Each converted MS is checked for `table.dat` + the `ANTENNA` subtable
    rather than `table.info`: casacure does not write that file for tables
    (casacore did) — tmolteno/casacure#23.
    """
    cli = Path(sys.executable).parent / "tart2ms"
    tmp = Path(tempfile.mkdtemp(prefix="tart2ms-release-"))
    try:
        local[str(cli)]["--json",
                        str(REPO / "test_data" / "data_2019_08_04_21_38_31_UTC.json"),
                        "--ms", str(tmp / "smoke_json.ms")] & FG
        local[str(cli)]["--hdf",
                        str(REPO / "test_data" / "vis_2026-06-12_04_25_46.149086.hdf"),
                        "--ms", str(tmp / "smoke_hdf.ms"),
                        "--rephase", "obs-midpoint", "--single-field"] & FG
        for ms in (tmp / "smoke_json.ms", tmp / "smoke_hdf.ms"):
            if not (ms / "table.dat").is_file() or not (ms / "ANTENNA").is_dir():
                raise RuntimeError(f"conversion did not produce an MS: {ms}")
        print("  smoke: JSON and HDF both produced a valid measurement set")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

@task
def version(c) -> None:
    """Show what the release would tag and where it would bump."""
    v = _read_version()
    tag = _highest_tag()
    base = _reconciled_version()
    nxt = _bump_patch(base)
    if tag and _version_key(tag) > _version_key(v):
        print(f"tart2ms: pyproject {v} is BEHIND {tag} — reconciling from {tag}")
    print(f"tart2ms: pyproject {v}  ->  would bump to {nxt} and tag v{nxt}")
    print(f"          CHANGES.md: ## Unreleased -> ## v{nxt} - <title>, "
          f"pyproject.toml + CITATION.cff stamped")
    print(f"          (or tag v{v} as-is with --no-bump; "
          f"--version X.Y.Z tags that exact version)")


@task
def test(c) -> None:
    """Run the pre-release gate: lint, smoke conversions, then the suite CI runs."""
    _run_flake8()
    _run_smoke_conversions()
    _run_pytest()
    print("=== test job PASS")


@task
def release(c,
            version: str | None = None,
            bump: bool = True,
            skip_tests: bool = False) -> None:
    """Run the full tart2ms release chain.

    1. run the pre-release gate (`invoke test`: flake8 fatal scan, the smoke
       conversions, then the same suite CI runs); skip with --skip-tests for
       an emergency re-tag only.
    2. bump the patch version in pyproject.toml and CITATION.cff and stamp
       CHANGES.md (the `## Unreleased` entries move under `## vX.Y.Z`;
       skip with --no-bump, or override with --version X.Y.Z which implies
       --no-bump since the tag is given explicitly) and commit them as
       `release: bump to X.Y.Z`.
    3. tag vX.Y.Z (annotated) and push the branch + tag; the tag push
       triggers deploy_module.yaml which publishes to PyPI.
    4. monitor: wait for the publish workflow run of the tag to go green,
       then wait for the version to appear on PyPI.

    Idempotent: a tag already on origin is verified and skipped, not re-pushed.
    """
    status(c)  # safety report first: stale tag fetch, unreachable tag, drift

    if skip_tests:
        print("=== --skip-tests: the local test gate is NOT run")
    else:
        test(c)

    # An explicit --version means "tag exactly this"; the tree's version must
    # not be touched or the tag and tree would drift apart.
    if version is not None:
        bump = False

    if bump:
        new = _bump_patch(_reconciled_version())
        if _tag_exists(f"v{new}"):
            raise RuntimeError(
                f"v{new} already exists on origin — your tag fetch is stale "
                f"(git fetch --tags); a released tag is never re-taken")
        print(f"=== bumping version to {new}")
        _bump_version(new)
        if _stamp_changelog(new):
            print(f"=== CHANGES.md: ## Unreleased -> ## v{new}")
        _git("add", "pyproject.toml", "CITATION.cff", "CHANGES.md")
        _git("commit", "-m", f"release: bump to {new}")

    v = version or _read_version()
    tag = f"v{v}"
    section = _changelog_section(v)
    if not section:
        print(f"    warning: CHANGES.md has no '## v{v}' section, so the tag "
              f"will be published without release notes (add one, or let the "
              f"default bump stamp it)")

    commit = _git("rev-parse", "HEAD").strip()
    if _tag_exists(tag):
        print(f"=== {tag} already on origin -- verified, skipping")
        return

    dirty = _git("status", "--porcelain").strip()
    if dirty:
        raise RuntimeError(
            "tree is dirty; commit the release (CHANGES.md, pyproject.toml, "
            f"CITATION.cff) first\n{dirty}")

    branch = _branch()
    print(f"=== tagging {tag} (commit {commit[:12]})")
    _git("tag", "-a", tag, "-m", tag)
    local["git"]["-C", str(REPO), "push", "origin", branch, tag] & FG

    print("=== monitoring CI/CD")
    runs = _wait_for_ci(tag)
    for name, run_id in runs.items():
        if run_id is not None:
            _watch_run(name, run_id)
    _wait_for_pypi(v)

    if section:
        notes = REPO / f".release-notes-{v}.md"
        notes.write_text(section + "\n")
        local["gh"]["release", "create", tag, "-R", REPO_SLUG,
                    "--title", f"tart2ms {tag}",
                    "--notes-file", str(notes)] & FG
        notes.unlink()
        print(f"=== GitHub release {tag} created")

    print(f"\n=== RELEASE COMPLETE: tart2ms {tag}")


@task
def status(c) -> None:
    """Everything that would make a release unsafe, in one screen."""
    version = _read_version()
    citation = re.search(r"(?m)^version: (\S+)", (REPO / "CITATION.cff").read_text())
    tag = _highest_tag()
    print(f"tart2ms: pyproject {version}   CITATION {citation.group(1) if citation else '?'}")
    print(f"         highest v* tag {tag or '(none)'}")

    if tag and _version_key(version) < _version_key(tag):
        print(f"  ! pyproject is behind {tag}: a patch release would go to "
              f"{_bump_patch(_reconciled_version())}")
    elif tag and _version_key(version) == _version_key(tag):
        print(f"   pyproject matches the tag; next release would be "
              f"{_bump_patch(_reconciled_version())}")
    else:
        print("   pyproject is ahead of every tag (unreleased work)")

    # A release tag that exists but is not on this branch's history means the
    # branch predates it: bumping straight from pyproject could re-take it.
    git = local["git"]["-C", str(REPO)]
    if tag:
        rc = git["merge-base", "--is-ancestor", tag, "HEAD"].run(retcode=None)[0]
        if rc != 0:
            print(f"  ! {tag} is NOT reachable from this branch — release from "
                  f"an up-to-date branch")

    rc, out, _ = git["rev-list", "--left-right", "--count", "@{u}...HEAD"].run(retcode=None)
    if rc == 0 and out.strip():
        behind, ahead = (out.split() + ["0", "0"])[:2]
        if int(behind):
            print(f"  ! branch is {behind} commit(s) behind its upstream — pull "
                  f"before releasing")
        elif int(ahead):
            print(f"   branch is {ahead} commit(s) ahead of its upstream "
                  f"(unpushed work)")
        else:
            print("   branch is in step with its upstream")

    lines = _git("status", "--porcelain").splitlines()
    dirty = [line for line in lines if not line.startswith("??")]
    untracked = [line for line in lines if line.startswith("??")]
    shown = ", ".join(dirty[:3]) + ("…" if len(dirty) > 3 else "")
    print(f"   modified tracked files: {len(dirty)}" + (f" ({shown})" if dirty else ""))
    if untracked:
        # release refuses on ANY porcelain line, untracked included, so a
        # stray handover.md would block the release if not reported here.
        names = ", ".join(line[3:] for line in untracked[:3])
        print(f"   untracked: {len(untracked)} ({names}) — release refuses a "
              f"dirty tree, so these must be committed, moved or ignored first")
    print(f"   interpreter {sys.version.split()[0]} (project venv); CI runs 3.11/3.12/3.13")
