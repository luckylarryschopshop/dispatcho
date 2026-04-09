"""Unit tests for router.py — path resolution, bash commands, local ops."""
from __future__ import annotations

import os
import pytest

from dispatcho.router import Router


@pytest.fixture
def router(tmp_path):
    """Router with a temp working directory."""
    return Router(working_dir=str(tmp_path))


class TestResolve:
    """Contract: _resolve() keeps paths within working_dir."""

    def test_resolve_relative(self, router, tmp_path):
        assert router._resolve("foo.txt") == str(tmp_path / "foo.txt")

    def test_resolve_dot(self, router, tmp_path):
        assert router._resolve(".") == str(tmp_path)

    def test_resolve_empty(self, router, tmp_path):
        assert router._resolve("") == str(tmp_path)

    def test_resolve_subdir(self, router, tmp_path):
        assert router._resolve("src/main.py").startswith(str(tmp_path))

    def test_resolve_rejects_escape(self, router):
        with pytest.raises(ValueError, match="escapes working directory"):
            router._resolve("/etc/passwd")

    def test_resolve_rejects_parent_without_flag(self, router):
        with pytest.raises(ValueError, match="escapes"):
            router._resolve("..")

    def test_resolve_allows_parent_with_flag(self, router):
        resolved = router._resolve("..", allow_parent=True)
        assert os.path.isabs(resolved)

    def test_resolve_rejects_non_string(self, router):
        with pytest.raises(TypeError):
            router._resolve(123)

    def test_resolve_parent_allowed_for_cd(self, router, tmp_path):
        """.. is only allowed with allow_parent=True (used by change_dir)."""
        resolved = router._resolve("..", allow_parent=True)
        assert os.path.isabs(resolved)


class TestBuildArgs:
    """Contract: _build_args() returns list (no shell interpretation)."""

    def test_git_status(self):
        args = Router._build_args("git status", {})
        assert args == ["git", "status", "--short"]

    def test_git_commit_message(self):
        args = Router._build_args("git commit", {"message": "fix bug"})
        assert args == ["git", "commit", "-m", "fix bug"]

    def test_git_commit_message_with_special_chars(self):
        """Shell metacharacters in message are safe because no shell."""
        args = Router._build_args("git commit", {"message": "fix; rm -rf /"})
        assert args == ["git", "commit", "-m", "fix; rm -rf /"]
        # This is safe because subprocess doesn't use shell

    def test_git_push(self):
        args = Router._build_args("git push", {"remote": "origin", "branch": "main"})
        assert args == ["git", "push", "origin", "main"]

    def test_git_init(self):
        args = Router._build_args("git init", {})
        assert args == ["git", "init"]


class TestExecuteContract:
    """Contract: execute() returns {success, result, method}."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_required_keys(self, router):
        result = await router.execute("list_files", {"path": "."})
        assert "success" in result
        assert "result" in result
        assert "method" in result

    @pytest.mark.asyncio
    async def test_unknown_intent(self, router):
        result = await router.execute("nonexistent", {})
        assert result["success"] is False
        assert "No route" in result["result"]

    @pytest.mark.asyncio
    async def test_invalid_intent_type(self, router):
        result = await router.execute(123, {})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_invalid_params_type(self, router):
        result = await router.execute("list_files", "not a dict")
        assert result["success"] is False


class TestLocalOps:
    """Contract: local filesystem operations work correctly."""

    @pytest.mark.asyncio
    async def test_create_and_read_file(self, router, tmp_path):
        result = await router.execute("create_file", {"path": "test.txt", "content": "hello"})
        assert result["success"]
        assert (tmp_path / "test.txt").read_text() == "hello"

        result = await router.execute("read_file", {"path": "test.txt"})
        assert result["success"]
        assert result["result"] == "hello"

    @pytest.mark.asyncio
    async def test_create_dir(self, router, tmp_path):
        result = await router.execute("create_dir", {"path": "subdir"})
        assert result["success"]
        assert (tmp_path / "subdir").is_dir()

    @pytest.mark.asyncio
    async def test_list_files(self, router, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = await router.execute("list_files", {"path": "."})
        assert result["success"]
        assert "a.txt" in result["result"]
        assert "b.txt" in result["result"]

    @pytest.mark.asyncio
    async def test_delete_file(self, router, tmp_path):
        (tmp_path / "temp.txt").write_text("delete me")
        result = await router.execute("delete_file", {"path": "temp.txt"})
        assert result["success"]
        assert not (tmp_path / "temp.txt").exists()

    @pytest.mark.asyncio
    async def test_move_file(self, router, tmp_path):
        (tmp_path / "old.txt").write_text("move me")
        result = await router.execute("move_file", {"source": "old.txt", "destination": "new.txt"})
        assert result["success"]
        assert not (tmp_path / "old.txt").exists()
        assert (tmp_path / "new.txt").read_text() == "move me"

    @pytest.mark.asyncio
    async def test_copy_file(self, router, tmp_path):
        (tmp_path / "src.txt").write_text("copy me")
        result = await router.execute("copy_file", {"source": "src.txt", "destination": "dst.txt"})
        assert result["success"]
        assert (tmp_path / "src.txt").exists()
        assert (tmp_path / "dst.txt").read_text() == "copy me"

    @pytest.mark.asyncio
    async def test_create_file_unescapes_newlines(self, router, tmp_path):
        result = await router.execute("create_file", {"path": "test.py", "content": "line1\\nline2"})
        assert result["success"]
        assert (tmp_path / "test.py").read_text() == "line1\nline2"

    @pytest.mark.asyncio
    async def test_search_files(self, router, tmp_path):
        (tmp_path / "readme.md").write_text("hello")
        (tmp_path / "app.py").write_text("world")
        result = await router.execute("search_files", {"pattern": "*.py"})
        assert result["success"]
        assert "app.py" in result["result"]

    @pytest.mark.asyncio
    async def test_read_missing_file(self, router):
        result = await router.execute("read_file", {"path": "nonexistent.txt"})
        assert result["success"]  # execute succeeds but result says not found
        assert "not found" in result["result"].lower() or "File not found" in result["result"]


class TestChangeDir:
    """Contract: change_dir updates working_dir."""

    @pytest.mark.asyncio
    async def test_change_to_subdir(self, router, tmp_path):
        (tmp_path / "sub").mkdir()
        result = await router.execute("change_dir", {"path": "sub"})
        assert result["success"]
        assert router.working_dir == str(tmp_path / "sub")

    @pytest.mark.asyncio
    async def test_change_to_parent(self, router, tmp_path):
        result = await router.execute("change_dir", {"path": ".."})
        assert result["success"]

    @pytest.mark.asyncio
    async def test_change_to_nonexistent(self, router):
        result = await router.execute("change_dir", {"path": "does_not_exist"})
        assert "not found" in result["result"].lower() or "Directory not found" in result["result"]
