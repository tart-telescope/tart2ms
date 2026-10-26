"""Release orchestration for tart2ms.

Following the example of ../casacure/tasks.py: `invoke release` runs the
pre-release gate locally before any tag is pushed to GitHub, then lets CI
publish and reports the outcome, so a broken release becomes a local failure
the operator sees first.

Usage (from the tart2ms repo root, with a venv that has invoke + plumbum):

    invoke version                      # show what the release would tag
    invoke test                         # the pre-release tests only
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
  1. pre-release tests: `python -m pytest tart2ms/tests/` in the current
     interpreter (the same suite CI runs; use `uv run invoke test` to get
     the project venv).
  2. version bump + CHANGES.md stamp + release commit (`release: bump to
     X.Y.Z`).
  3. annotated tag vX.Y.Z and `git push origin <branch> vX.Y.Z`; the tag
     push triggers `.github/workflows/deploy_module.yaml` which builds and
     publishes the package to PyPI.  (The `testing.yaml` suite runs on the
     push of the release commit as well.)
  4. monitor: wait for every workflow run of the tag to register and go
     green, then poll PyPI until the new version is live.

Idempotent: a tag already on origin is verified and skipped, not re-pushed.
Requires: invoke, plumbum (pinned in the dev dependency group), git, gh
(authenticated), and network access for the CI/PyPI polls.
"""
import json
import re
import sys
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


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

@task
def version(c) -> None:
    """Show what the release would tag and where it would bump."""
    v = _read_version()
    nxt = _bump_patch(v)
    print(f"tart2ms: pyproject {v}  ->  would bump to {nxt} and tag v{nxt}")
    print(f"          CHANGES.md: ## Unreleased -> ## v{nxt} - <title>, "
          f"pyproject.toml + CITATION.cff stamped")
    print(f"          (or tag v{v} as-is with --no-bump; "
          f"--version X.Y.Z tags that exact version)")


@task
def test(c) -> None:
    """Run the pre-release test suite (what CI runs on the push)."""
    _run_pytest()
    print("=== test job PASS")


@task
def release(c,
            version: str | None = None,
            bump: bool = True,
            skip_tests: bool = False) -> None:
    """Run the full tart2ms release chain.

    1. run the pre-release tests (`invoke test`, the same suite CI runs);
       skip with --skip-tests for an emergency re-tag only.
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
    if skip_tests:
        print("=== --skip-tests: the local test gate is NOT run")
    else:
        test(c)

    # An explicit --version means "tag exactly this"; the tree's version must
    # not be touched or the tag and tree would drift apart.
    if version is not None:
        bump = False

    if bump:
        new = _bump_patch(_read_version())
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

    status = _git("status", "--porcelain").strip()
    if status:
        raise RuntimeError(
            "tree is dirty; commit the release (CHANGES.md, pyproject.toml, "
            f"CITATION.cff) first\n{status}")

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
