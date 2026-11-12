"""Published git ref verifier and ephemeral read-only detached worktree manager."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

from beast_harness.types import RefVerificationError, VerifiedRepoRef, ensure_directory

_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class PublishedRefVerifier:
    def __init__(self, worktree_root: Path | None = None) -> None:
        default_root = Path.home() / ".beast-harness" / "worktrees"
        self.worktree_root = Path(worktree_root) if worktree_root is not None else default_root

    def _git_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    def _bare_cache_dir(self, repo_url: str) -> Path:
        slug = hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:20]
        cache_dir = ensure_directory(self.worktree_root / ".bare-cache") / f"{slug}.git"
        if not cache_dir.exists():
            subprocess.run(
                ["git", "init", "--bare", str(cache_dir)],
                check=True,
                capture_output=True,
                text=True,
                env=self._git_env(),
            )
        return cache_dir

    def _resolve_remote_sha(self, repo_url: str, ref: str) -> str:
        cleaned_url = repo_url.strip()
        cleaned_ref = ref.strip()
        if not cleaned_url or not cleaned_ref:
            raise RefVerificationError("Both repo_url and ref must be non-empty.")

        try:
            proc = subprocess.run(
                ["git", "ls-remote", cleaned_url],
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
                env=self._git_env(),
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RefVerificationError(
                f"Failed to query remote refs for {cleaned_url!r}: {exc}"
            ) from exc

        refs_map: dict[str, str] = {}
        advertised_shas: set[str] = set()
        for line in proc.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                sha, refname = parts[0].lower(), parts[1]
                refs_map[refname] = sha
                advertised_shas.add(sha)

        candidate_keys = (
            cleaned_ref,
            f"refs/heads/{cleaned_ref}",
            f"refs/tags/{cleaned_ref}^{{}}",
            f"refs/tags/{cleaned_ref}",
        )
        for key in candidate_keys:
            if key in refs_map:
                return refs_map[key]

        if _HEX40_RE.match(cleaned_ref):
            target_sha = cleaned_ref.lower()
            if target_sha in advertised_shas:
                return target_sha
            # Verify whether the exact 40-char SHA is published/reachable on the remote
            cache_dir = self._bare_cache_dir(cleaned_url)
            fetch = subprocess.run(
                ["git", "--git-dir", str(cache_dir), "fetch", "--depth=1", cleaned_url, target_sha],
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
                env=self._git_env(),
            )
            if fetch.returncode == 0:
                return target_sha
            raise RefVerificationError(
                f"Commit {target_sha} is not published or reachable on remote {cleaned_url!r}."
            )

        raise RefVerificationError(
            f"Reference {cleaned_ref!r} was not found on remote {cleaned_url!r}."
        )

    def verify_published_ref(
        self,
        repo_url: str,
        ref: str = "HEAD",
        *,
        base_ref: str | None = None,
    ) -> VerifiedRepoRef:
        commit_sha = self._resolve_remote_sha(repo_url, ref)
        base_sha = self._resolve_remote_sha(repo_url, base_ref) if base_ref else None
        return VerifiedRepoRef(
            repo_url=repo_url.strip(),
            requested_ref=ref.strip(),
            commit_sha=commit_sha,
            base_sha=base_sha,
        )

    def materialize_readonly_worktree(
        self,
        verified_ref: VerifiedRepoRef,
        *,
        session_id: str,
    ) -> VerifiedRepoRef:
        if not _SAFE_SESSION_RE.match(session_id):
            raise RefVerificationError(f"Unsafe session_id for worktree path: {session_id!r}")

        cache_dir = self._bare_cache_dir(verified_ref.repo_url)
        try:
            subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(cache_dir),
                    "fetch",
                    "--depth=1",
                    verified_ref.repo_url,
                    verified_ref.commit_sha,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60.0,
                env=self._git_env(),
            )
        except subprocess.SubprocessError as exc:
            raise RefVerificationError(
                f"Failed to fetch published commit {verified_ref.commit_sha}: {exc}"
            ) from exc

        wt_parent = ensure_directory(self.worktree_root / "sessions")
        wt_dir = wt_parent / session_id
        if wt_dir.exists():
            self.cleanup_worktree(wt_dir)

        try:
            subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(cache_dir),
                    "worktree",
                    "add",
                    "--detach",
                    str(wt_dir),
                    verified_ref.commit_sha,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
                env=self._git_env(),
            )
        except subprocess.SubprocessError as exc:
            raise RefVerificationError(
                f"Failed to materialize detached worktree for {verified_ref.commit_sha}: {exc}"
            ) from exc

        # Strip write bits across files first, then directories bottom-up so OS enforces read-only
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        for root, dirs, files in os.walk(wt_dir, topdown=False):
            root_path = Path(root)
            for fname in files:
                fpath = root_path / fname
                if not fpath.is_symlink():
                    mode = fpath.stat().st_mode
                    fpath.chmod(mode & ~write_bits)
            for dname in dirs:
                dpath = root_path / dname
                if not dpath.is_symlink():
                    mode = dpath.stat().st_mode
                    dpath.chmod(mode & ~write_bits)
        wt_dir.chmod(wt_dir.stat().st_mode & ~write_bits)

        return verified_ref.model_copy(update={"local_worktree_path": str(wt_dir)})

    def cleanup_worktree(self, worktree_path: str | Path | None) -> None:
        if worktree_path is None:
            return
        wt_dir = Path(worktree_path)
        if not wt_dir.exists():
            return

        # Restore owner write permissions top-down so rmtree can delete cleanly
        wt_dir.chmod(wt_dir.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        for root, dirs, files in os.walk(wt_dir, topdown=True):
            root_path = Path(root)
            for dname in dirs:
                dpath = root_path / dname
                if not dpath.is_symlink():
                    dpath.chmod(dpath.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            for fname in files:
                fpath = root_path / fname
                if not fpath.is_symlink():
                    fpath.chmod(fpath.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR)

        shutil.rmtree(wt_dir, ignore_errors=False)

        bare_cache_root = self.worktree_root / ".bare-cache"
        if bare_cache_root.exists():
            for cache_repo in bare_cache_root.glob("*.git"):
                subprocess.run(
                    ["git", "--git-dir", str(cache_repo), "worktree", "prune"],
                    check=False,
                    capture_output=True,
                    env=self._git_env(),
                )
