"""
Release automation for tart2ms: invoke tasks backed by plumbum commands.

    uv run invoke --list
    uv run invoke test                  # exhaustive local test run
    uv run invoke test --quick          # one interpreter, no package build
    uv run invoke bump --dry-run        # show the next version (and why)
    uv run invoke status                # version vs tag vs CI state
    uv run invoke release               # test -> bump -> CI publish (prompts)
    uv run invoke release --dry-run     # print the plan, change nothing

A release is self-contained: three stages, all inside this repo.

    1. test    uv sync --dev + uv lock --check, the flake8 fatal gate scoped
               to the package, the informational style count, pytest on every
               supported interpreter (3.11/3.12/3.13, matching
               requires-python and .github/workflows/testing.yaml), CLI smoke
               tests that actually convert the JSON *and* HDF fixtures into
               measurement sets, and a package build.
    2. bump    the next version, pyproject.toml + CITATION.cff together, and
               a CHANGES.md section for it.
    3. publish commit + annotated tag v<version> + push -> .github/workflows/
               deploy_module.yaml builds and publishes to PyPI with
               pypa/gh-action-pypi-publish (PYPI_TART2MS_TOKEN) -> wait for
               that run to go green.

Shipping the release inside the TART tools image is deliberately *not* part
of this chain (see RELEASE.md): the image installs `tart2ms>=…` from PyPI, so
rebuilding it is a separate, explicit step — `invoke -r
~/tart-telescope/tools release`, normally driven by disko's release chain.

Version reconciliation, because this repo has been inconsistent: the tag and
PyPI are at 0.9.4 while pyproject.toml still says 0.9.3 (and CHANGES.md never
got a v0.9.4 section). The next version is therefore computed as the bump of
**whichever is higher — the tag or the file** — so a release here goes to
0.9.5, never to an already-taken 0.9.4.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from invoke import Exit, task
from plumbum import colors, local

# ---------------------------------------------------------------- locations

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGES = ROOT / "CHANGES.md"
CITATION = ROOT / "CITATION.cff"
TEST_DATA = ROOT / "test_data"

# Same interpreters requires-python = ">=3.11,<3.14" allows, and the same
# ones .github/workflows/testing.yaml builds against after this change.
PYTHON_MATRIX = ("3.11", "3.12", "3.13")

# Lint the package only: the repo carries gitignored venv/ and .venv/ trees
# (and a .pytest_cache) that `flake8 .` would otherwise walk — that is how
# this file's own author first saw 598 "fatal" findings that were all
# site-packages.
LINT_SCOPE = "tart2ms"

UV = local["uv"]
GIT = local["git"]

_LEVELS = ("patch", "minor", "major")
# Matches version = '0.9.3' (single quotes, as pyproject uses) and "0.9.3".
_VERSION_RE = re.compile(r"""^(?P<prefix>version\s*=\s*)(?P<q>['"])(?P<v>[^'"]+)(?P=q)""", re.M)
_CITATION_VERSION_RE = re.compile(r"""^(?P<prefix>version:\s*)(?P<v>\S+)""", re.M)


# ---------------------------------------------------------------- helpers

def _run(cmd, **kwargs):
    """Run a plumbum command from the repo root, echoing it, failing on error."""
    print(_light_blue(f"+ {cmd}"), flush=True)
    kwargs.setdefault("cwd", str(ROOT))  # the chain may be invoked from anywhere
    return cmd.run(**kwargs)


def _capture(cmd):
    """Run a plumbum command quietly from the repo root -> (rc, out, err)."""
    return cmd.run(retcode=None, cwd=str(ROOT))


def _paint(style, text: str) -> str:
    """Apply a plumbum style portably: plumbum 2 dropped ``colors.x("text")``
    in favour of ``text | style``, which also degrades to plain text when
    stdout is not a tty — what we want in captured CI logs."""
    return text | style


def _bold(text: str) -> str:
    return _paint(colors.bold, text)


def _light_blue(text: str) -> str:
    return _paint(colors.light_blue, text)


def _green(text: str) -> str:
    return _paint(colors.green, text)


def _yellow(text: str) -> str:
    return _paint(colors.yellow, text)


def _banner(text: str) -> None:
    print(_bold(f"\n=== {text}"), flush=True)


def _git(*args):
    rc, out, err = _capture(GIT[list(args)])
    if rc != 0:
        raise Exit(f"git {' '.join(args)} failed: {err.strip()}")
    return out.strip()


def _confirm(message: str, assume_yes: bool) -> None:
    if assume_yes:
        print(_yellow(f"{message} (--yes: proceeding)"))
        return
    if input(f"{message} [y/N] ").strip().lower() not in ("y", "yes"):
        raise Exit("aborted")


# ------------------------------------------------------------ version logic

def read_version() -> str:
    match = _VERSION_RE.search(PYPROJECT.read_text())
    if not match:
        raise Exit(f"no version = ... line in {PYPROJECT}")
    return match.group("v")


def write_version(version: str) -> None:
    text = PYPROJECT.read_text()
    PYPROJECT.write_text(_VERSION_RE.sub(lambda m: f"{m.group('prefix')}{m.group('q')}{version}{m.group('q')}", text, count=1))


def read_citation_version() -> str | None:
    match = _CITATION_VERSION_RE.search(CITATION.read_text()) if CITATION.exists() else None
    return match.group("v") if match else None


def write_citation_version(version: str) -> bool:
    if not CITATION.exists():
        return False
    text = CITATION.read_text()
    if _CITATION_VERSION_RE.search(text) is None:
        return False
    CITATION.write_text(_CITATION_VERSION_RE.sub(lambda m: f"{m.group('prefix')}{version}", text, count=1))
    return True


def latest_tag() -> str:
    """Highest v* tag **anywhere in the repository**, not just on this branch.

    `git describe` only sees tags reachable from HEAD, and releases cut from
    another branch are invisible to it: v0.9.4 lives on master while
    casacure-backend predates it, so a describe-based bump would have tried
    to re-take an existing tag. Prerelease tags (v0.7.0rc1, v0.2.1b3) parse
    to their numeric core so they never break the comparison.
    """
    rc, out, _ = _capture(GIT["tag", "-l", "v*"])
    tags = out.split() if rc == 0 else []
    return max(tags, key=_key) if tags else ""


def _key(version: str) -> tuple[int, int, int]:
    """(major, minor, patch) from any version-ish string; (0,0,0) if none."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
    return tuple(int(group) for group in match.groups()) if match else (0, 0, 0)


def next_version(level: str) -> str:
    """Bump whichever is higher: pyproject's version or the latest tag.

    Tag v0.9.4 and PyPI 0.9.4 exist while pyproject says 0.9.3, so a naive
    patch bump would try to re-release 0.9.4. Taking the max lands on 0.9.5.
    """
    if level not in _LEVELS:
        raise Exit(f"--level must be one of {_LEVELS}, got {level!r}")
    current = read_version()
    tag = latest_tag().lstrip("v")
    base = max(_key(current), _key(tag)) if tag else _key(current)
    major, minor, patch = base
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _changes_for(version: str) -> None:
    """Reuse the section if one exists (entries are written as work lands),
    otherwise draft `## v<version> — …` from git history and say so."""
    text = CHANGES.read_text()
    if re.search(rf"^## v?{re.escape(version)}\b", text, re.M):
        print(f"  CHANGES.md already documents {version}")
        return
    rc, out, _ = _capture(GIT["log", f"{latest_tag()}..HEAD", "--pretty=%s"])
    bullets = "\n".join(f"- {line}" for line in out.splitlines() if line.strip())
    section = f"## v{version} — Release v{version}\n\n{bullets or '- Maintenance release.'}\n"
    head, sep, rest = text.partition("\n")
    CHANGES.write_text(f"{head}\n{sep}{section}\n{rest.lstrip()}\n")
    print(_yellow(f"  CHANGES.md: drafted a `## v{version}` section from git history — please review"))


# ------------------------------------------------------------------ guards

def _assert_clean_for_publish() -> None:
    """Only the three files a release edits may be *modified*.

    Untracked files (test .ms outputs, scratch) are reported but do not block:
    they are not part of what the tag points at.
    """
    allowed = {PYPROJECT, CHANGES, CITATION}
    unexpected, untracked = [], []
    for line in _git("status", "--porcelain").splitlines():
        if not line.strip():
            continue
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if line.startswith("??"):
            untracked.append(line)
        elif (ROOT / path) not in allowed:
            unexpected.append(line)
    if untracked:
        print(_yellow("  untracked, not part of the release:"))
        for line in untracked[:10]:
            print(_yellow(f"    {line}"))
        if len(untracked) > 10:
            print(_yellow(f"    … and {len(untracked) - 10} more"))
    if unexpected:
        raise Exit(
            "working tree has unrelated changes; commit or stash them first:\n  "
            + "\n  ".join(unexpected)
        )


# ------------------------------------------------------------------- tasks

@task(help={"quick": "single interpreter and no package build"})
def test(c, quick=False):
    """Exhaustive local tests — everything a release must pass first."""
    _banner("test: environment")
    _run(UV["sync", "--dev"])
    _run(UV["lock", "--check"])  # uv.lock agrees with pyproject.toml

    _banner("test: lint (fatal findings only, package scope)")
    _run(
        UV[
            "run", "flake8", LINT_SCOPE, "--count",
            "--select=E9,F63,F7,F82", "--show-source", "--statistics",
        ]
    )
    _, out, _ = _capture(
        UV[
            "run", "flake8", LINT_SCOPE, "--count", "--exit-zero",
            "--max-complexity=10", "--max-line-length=127", "--statistics",
        ]
    )
    total = (out.strip().splitlines() or ["?"])[-1]
    print(_yellow(f"  style findings (informational, not fatal): {total}"))

    for version in (PYTHON_MATRIX[:1] if quick else PYTHON_MATRIX):
        _banner(f"test: pytest on python {version}")
        _run(UV["run", "--python", version, "pytest"])

    _banner("test: CLI smoke tests (both fixture formats -> real MS)")
    tmp = Path(tempfile.mkdtemp(prefix="tart2ms-release-"))
    try:
        _run(UV["run", "tart2ms", "--help"])
        # JSON fixture (the Makefile's `testms` target)
        json_ms = tmp / "smoke_json.ms"
        _run(
            UV[
                "run", "tart2ms", "--json",
                str(TEST_DATA / "data_2019_08_04_21_38_31_UTC.json"),
                "--ms", str(json_ms),
            ]
        )
        # HDF fixture (the Makefile's `h5` target)
        hdf_ms = tmp / "smoke_hdf.ms"
        _run(
            UV[
                "run", "tart2ms", "--hdf",
                str(TEST_DATA / "vis_2026-06-12_04_25_46.149086.hdf"),
                "--ms", str(hdf_ms), "--rephase", "obs-midpoint", "--single-field",
            ]
        )
        for ms in (json_ms, hdf_ms):
            # Check table.dat + a subtable rather than table.info: casacure
            # (the Rust casacore replacement this branch writes through) does
            # NOT create table.info, while python-casacore did — the tracked
            # fixtures have one, freshly converted MS directories do not.
            if not (ms / "table.dat").is_file() or not (ms / "ANTENNA").is_dir():
                raise Exit(f"conversion did not produce a measurement set: {ms}")
        print("  both conversions produced a valid MS (table.dat + ANTENNA present)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not quick:
        _banner("test: package build")
        shutil.rmtree(ROOT / "dist", ignore_errors=True)
        _run(UV["build"])

    print(_green("\ntest: OK"))


@task(
    help={
        "level": "patch | minor | major (default: patch)",
        "dry_run": "show what would change without writing anything",
    }
)
def bump(c, level="patch", dry_run=False):
    """Set the next version in pyproject + CITATION and open a CHANGES section."""
    _banner("bump")
    current, tag, new = read_version(), latest_tag(), next_version(level)

    if tag and _key(current) < _key(tag.lstrip("v")):
        print(_yellow(f"  pyproject ({current}) is behind tag {tag} — releasing from the tag, so"))
        print(_yellow(f"  the next version is {new} (a naive patch bump would re-take {tag.lstrip('v')})"))
    print(f"  version {current} -> {new} ({level})")
    print(f"  last tag: {tag or '(none)'}")
    print(f"  CITATION.cff: {read_citation_version() or '(no version line)'}")
    rc, out, _ = _capture(GIT["log", f"{tag}..HEAD" if tag else "-10", "--pretty=%s"])
    for line in [line for line in out.splitlines() if line.strip()][:12]:
        print(f"    - {line}")

    if dry_run:
        print(_yellow("  dry run: no files written"))
        return

    write_version(new)
    if write_citation_version(new):
        print(f"  CITATION.cff -> {new}")
    _changes_for(new)
    print(_green(f"  pyproject.toml -> {new}"))


@task(
    help={
        "yes": "skip the confirmation prompt",
        "wait": "wait for the deploy_module publish workflow (default)",
    }
)
def publish(c, yes=False, wait=True):
    """Commit the bump, tag vX.Y.Z and push: CI publishes to PyPI.

    .github/workflows/deploy_module.yaml triggers on the v*.*.* tag, runs
    `python -m build` and uploads with pypa/gh-action-pypi-publish
    (PYPI_TART2MS_TOKEN), so this never runs twine locally.
    """
    _banner("publish")
    version = read_version()
    tag = f"v{version}"

    if tag in _git("tag").splitlines():
        raise Exit(f"{tag} already exists — run `invoke bump` first")
    _assert_clean_for_publish()

    _confirm(f"Release {version}: commit, tag {tag} and push (CI publishes to PyPI)?", yes)

    _run(GIT["add", str(PYPROJECT), str(CHANGES), str(CITATION)])
    if _capture(GIT["diff", "--cached", "--quiet"])[0] != 0:
        _run(GIT["commit", "-m", f"Release {version}"])
    else:
        print("  bump already committed; tagging the current commit")
    _run(GIT["tag", "-a", tag, "-m", f"Release {version}"])
    _run(GIT["push", "--follow-tags", "origin", "HEAD"])

    if wait:
        _wait_for_publish(tag)
    else:
        print(_yellow("  not waiting for CI; verify the deploy_module run yourself"))

    print(_green(f"\n{tag} pushed — PyPI publish is CI's job from here."))
    print(_bold("  next (separate step, not run automatically):"))
    print("  to ship this release in the TART tools image, rebuild it —")
    print("  `invoke -r ~/tart-telescope/tools release` (its tart2ms pin is")
    print("  a loose >=, so no Dockerfile edit is needed) and then release")
    print("  tart_cargo with that tag. See RELEASE.md.")


def _wait_for_publish(tag: str, timeout_s: int = 1800, poll_s: int = 20) -> None:
    # plumbum's local.which() raises when a program is missing and hands back
    # a path (not a command) when it exists, so detect with shutil.which and
    # build the command with local[...].
    if shutil.which("gh") is None:
        raise Exit("gh CLI is required to wait for the PyPI publish (or pass --no-wait)")
    gh = local["gh"]

    print(_light_blue(f"  waiting for the deploy_module run on {tag} ..."))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rc, out, _ = _capture(
            gh[
                "run", "list", "--repo", "tart-telescope/tart2ms", "--workflow",
                "deploy_module.yaml", "--limit", "10",
                "--json", "headBranch,status,conclusion,url",
            ]
        )
        if rc == 0 and out.strip() and out.strip() != "[]":
            for run in json.loads(out):
                if run.get("headBranch") == tag and run.get("status") == "completed":
                    if run.get("conclusion") == "success":
                        print(_green(f"  PyPI publish OK: {run['url']}"))
                        return
                    raise Exit(f"publish workflow failed: {run['url']}")
        time.sleep(poll_s)
    raise Exit(f"timed out after {timeout_s}s waiting for the publish run on {tag}")


@task(
    help={
        "level": "patch | minor | major (default: patch)",
        "yes": "skip the confirmation prompt",
        "wait": "wait for the PyPI publish workflow (default)",
        "dry_run": "print the full plan and exit without running anything",
    }
)
def release(c, level="patch", yes=False, wait=True, dry_run=False):
    """Full release: exhaustive tests -> version bump -> CI publish."""
    current, planned = read_version(), next_version(level)

    if dry_run:
        _banner("tart2ms release plan (dry run)")
        print(f"  1. test     sync + lock check, flake8 gate, pytest on {', '.join(PYTHON_MATRIX)},")
        print("              JSON + HDF conversion smoke tests, uv build")
        print(f"  2. bump     {current} -> {planned} ({level}; highest of tag/latest vs pyproject)")
        print("              pyproject.toml + CITATION.cff + CHANGES.md section")
        print(f"  3. publish  commit + tag v{planned} + push -> deploy_module.yaml -> PyPI, wait")
        print(_yellow("  not run: the tools image rebuild (separate, explicit — see RELEASE.md)"))
        print(_yellow("  dry run: nothing executed"))
        return

    _confirm(f"Release tart2ms {current} -> {planned} (test, bump, tag, CI publish)?", yes)
    yes = True  # confirmed once

    test(c, quick=False)
    bump(c, level=level, dry_run=False)
    publish(c, yes=yes, wait=wait)


@task
def status(c):
    """Show version, tags, and anything that would make a release unsafe."""
    _banner("status")
    current, tag, citation = read_version(), latest_tag(), read_citation_version()
    print(f"  pyproject  {current}")
    print(f"  CITATION   {citation or '(no version line)'}")
    print(f"  highest v* tag: {tag or '(none)'}")

    if tag and _key(current) < _key(tag):
        print(_yellow(f"  pyproject is BEHIND {tag}: a patch release would go to {next_version('patch')}"))
    elif tag and _key(current) == _key(tag):
        print(_green(f"  pyproject matches the tag; next release would be {next_version('patch')}"))
    else:
        print(_yellow(f"  pyproject is ahead of every tag (unreleased work); release as {current}"))

    # Releasing from a branch that trails the release tag or its upstream is
    # how versions get re-taken: report both conditions.
    if tag:
        rc, _, _ = _capture(GIT["merge-base", "--is-ancestor", tag, "HEAD"])
        if rc != 0:
            print(_yellow(f"  {tag} is NOT reachable from this branch — cut releases from an up-to-date branch"))
    rc, out, _ = _capture(GIT["rev-list", "--left-right", "--count", "@{u}...HEAD"])
    if rc == 0 and out.strip():
        behind, ahead = (out.split() + ["0", "0"])[:2]
        if int(behind):
            print(_yellow(f"  branch is {behind} commit(s) behind its upstream — update before releasing"))
        elif int(ahead):
            print(f"  branch is {ahead} commit(s) ahead of its upstream (unpushed work)")
        else:
            print("  branch is in step with its upstream")

    dirty = [line for line in _git("status", "--porcelain").splitlines() if not line.startswith("??")]
    print(f"  modified tracked files: {len(dirty)}")
    print(f"  python matrix: {', '.join(PYTHON_MATRIX)} (requires-python >=3.11,<3.14)")
