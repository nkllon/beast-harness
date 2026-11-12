"""Milestone 1 acceptance tests: Real git ref verification, secrets, and SQLite ledger."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from beast_harness.ledger import SessionLedger
from beast_harness.ref_verifier import PublishedRefVerifier
from beast_harness.secrets import (
    CallbackSecretProvider,
    ChainedSecretProvider,
    CommandSecretProvider,
    EnvSecretProvider,
)
from beast_harness.types import (
    DriverRunResult,
    ExecutionPosture,
    PostureViolationError,
    RefVerificationError,
    RunState,
    SecretResolutionError,
)


def _git(args: list[str], cwd: Path) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


@pytest.fixture()
def real_git_remote_and_clone(tmp_path: Path) -> tuple[Path, Path, str, str]:
    remote = tmp_path / "origin.git"
    remote.mkdir()
    _git(["init", "--bare", "-b", "main"], remote)

    clone = tmp_path / "working"
    subprocess.run(["git", "clone", str(remote), str(clone)], check=True, capture_output=True)
    _git(["config", "user.email", "test@energration.com"], clone)
    _git(["config", "user.name", "Test Operator"], clone)

    (clone / "module.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    _git(["add", "module.py"], clone)
    _git(["commit", "-m", "initial published commit"], clone)
    _git(["push", "-u", "origin", "main"], clone)
    published_sha = _git(["rev-parse", "HEAD"], clone)

    # Create a second UNPUSHED local commit
    (clone / "module.py").write_text("def add(a: int, b: int) -> int:\n    return a - b\n")
    _git(["add", "module.py"], clone)
    _git(["commit", "-m", "unpushed local mutation"], clone)
    unpushed_sha = _git(["rev-parse", "HEAD"], clone)

    return remote, clone, published_sha, unpushed_sha


def test_published_ref_verifier_accepts_remote_sha_and_rejects_unpushed_commit(
    tmp_path: Path,
    real_git_remote_and_clone: tuple[Path, Path, str, str],
):
    remote, _clone, published_sha, unpushed_sha = real_git_remote_and_clone
    verifier = PublishedRefVerifier(worktree_root=tmp_path / "worktrees")

    # Resolving branch 'main' resolves to exact 40-char published_sha
    verified = verifier.verify_published_ref(str(remote), "main")
    assert verified.commit_sha == published_sha

    # Resolving explicit published 40-char SHA succeeds
    verified_exact = verifier.verify_published_ref(str(remote), published_sha)
    assert verified_exact.commit_sha == published_sha

    # Unpushed commit SHA must fail closed!
    with pytest.raises(RefVerificationError):
        verifier.verify_published_ref(str(remote), unpushed_sha)

    # Materialize detached read-only worktree and verify writes are blocked by OS permissions
    mounted = verifier.materialize_readonly_worktree(verified, session_id="sess-ro-1")
    assert mounted.local_worktree_path is not None
    wt = Path(mounted.local_worktree_path)
    assert (wt / "module.py").read_text().strip().endswith("return a + b")

    with pytest.raises(PermissionError):
        (wt / "module.py").write_text("tampered")

    verifier.cleanup_worktree(wt)
    assert not wt.exists()


def test_chained_secret_provider_priority_and_redaction():
    secret_val = "crsr_super_secret_token_999"
    chain = ChainedSecretProvider(
        [
            CallbackSecretProvider(lambda k: "magpie_override" if k == "MAGPIE_KEY" else None),
            CommandSecretProvider({"CMD_KEY": ["python3", "-c", "print('from_cmd')"]}),
            EnvSecretProvider({"CURSOR_API_KEY": secret_val}),
        ]
    )
    assert chain.require_secret("MAGPIE_KEY") == "magpie_override"
    assert chain.require_secret("CMD_KEY") == "from_cmd"
    assert chain.require_secret("CURSOR_API_KEY") == secret_val
    assert secret_val not in repr(chain)
    assert secret_val not in str(chain)

    with pytest.raises(SecretResolutionError):
        chain.require_secret("MISSING_KEY")


def test_session_ledger_durability_and_independent_review_immutability(tmp_path: Path):
    db_file = tmp_path / "sessions.sqlite3"
    ledger1 = SessionLedger(db_file)

    initial = DriverRunResult(
        native_agent_id="bc-abc123",
        run_id="run-001",
        state=RunState.RUNNING,
    )
    rec = ledger1.create_session(
        session_id="sess-review-1",
        endpoint_id="cursor-cloud",
        driver_id="cursor-cloud",
        repo_ref=None,
        posture=ExecutionPosture.OBSERVE_ONLY,
        independent_review=True,
        facets={"extra_repos": []},
        initial_run=initial,
    )
    assert rec.status == RunState.RUNNING

    # Simulate MCP server restart by opening a fresh SessionLedger instance on the same file
    ledger2 = SessionLedger(db_file)
    loaded = ledger2.get_session("sess-review-1")
    assert loaded.native_agent_id == "bc-abc123"
    assert loaded.independent_review is True

    # Independent review sessions MUST block follow-up prompts fail-closed
    with pytest.raises(PostureViolationError):
        ledger2.assert_followup_allowed("sess-review-1")

    # Updating to terminal state records answer and deterministic audit_digest
    finished = DriverRunResult(
        native_agent_id="bc-abc123",
        run_id="run-001",
        state=RunState.FINISHED,
        answer="No blocking defects found.",
        structured_output={"verdict": "approve"},
        audit_events=[{"tool": "view_file", "path": "module.py"}],
    )
    updated = ledger2.update_from_driver_result("sess-review-1", finished)
    assert updated.status == RunState.FINISHED
    assert updated.answer == "No blocking defects found."
    assert updated.structured_output == {"verdict": "approve"}
    assert updated.audit_digest and len(updated.audit_digest) == 64
