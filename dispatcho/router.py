"""Router — executes classified intents via local Python or bash.

Filesystem operations use Python stdlib (os, shutil).
Git operations use subprocess (git CLI).

Contracts:
- execute() always returns {success: bool, result: str, method: str}
- All paths resolved via _resolve() and validated to stay within working_dir
- Bash commands use subprocess with timeout, no shell injection via user params
- Git commands parameterized (not string-interpolated from user input)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

# Intent → execution method
# "local:" = Python stdlib (os, shutil)
# "bash:" = subprocess (git CLI)
# "internal:" = state changes (change_dir)
INTENT_ROUTES = {
    # Filesystem
    "list_files":    "local:list_files",
    "create_file":   "local:create_file",
    "read_file":     "local:read_file",
    "edit_file":     "local:edit_file",
    "move_file":     "local:move_file",
    "copy_file":     "local:copy_file",
    "delete_file":   "local:delete_file",
    "create_dir":    "local:create_dir",
    "search_files":  "local:search_files",
    "current_dir":   "local:current_dir",

    # Git
    "git_status":    "bash:git status",
    "git_add":       "bash:git add",
    "git_commit":    "bash:git commit",
    "git_diff":      "bash:git diff",
    "git_log":       "bash:git log --oneline -20",
    "git_branch":    "bash:git branch",

    # Change directory — internal (updates working_dir)
    "change_dir":    "internal:change_dir",

    # Git — bash fallback (not in MCP git server)
    "git_init":      "bash:git init",
    "git_push":      "bash:git push",
    "git_pull":      "bash:git pull",
    "git_merge":     "bash:git merge",
    "git_clone":     "bash:git clone",

    # Dev tools
    "run_tests":     "bash:pytest",
    "install_deps":  "bash:pip install",
    "run_server":    "bash:uvicorn",
    "format_code":   "bash:black",
    "open_url":      "bash:open",
}


class Router:
    """Executes tool operations via local Python (filesystem) or subprocess (git).

    Usage:
        router = Router("/tmp/project")
        result = await router.execute("list_files", {"path": "."})
    """

    def __init__(self, working_dir: str = "."):
        self.working_dir = os.path.realpath(os.path.abspath(working_dir))
        self._root = self.working_dir  # cd cannot go above this
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass  # no async resources to clean up

    async def execute(self, intent: str, params: dict) -> dict:
        """Execute a classified intent.

        Contract:
        - REQUIRES: intent is str, params is dict
        - ENSURES: returns {success: bool, result: str, method: str}
        - ENSURES: method is one of "local", "bash", "internal", "none"
        """
        if not isinstance(intent, str):
            return {"success": False, "result": f"intent must be str", "method": "none"}
        if not isinstance(params, dict):
            return {"success": False, "result": f"params must be dict", "method": "none"}

        route = INTENT_ROUTES.get(intent)
        if not route:
            return {"success": False, "result": f"No route for intent: {intent}", "method": "none"}

        method, tool = route.split(":", 1)

        try:
            if method == "bash":
                result = self._call_bash(tool, params)
            elif method == "internal":
                result = self._call_internal(tool, params)
            elif method == "local":
                result = self._call_local(tool, params)
            else:
                return {"success": False, "result": f"Unknown method: {method}", "method": method}

            return {"success": True, "result": result or "(empty)", "method": method}
        except (FileNotFoundError, PermissionError, OSError) as e:
            return {"success": False, "result": f"OS error: {e}", "method": method}
        except subprocess.TimeoutExpired:
            return {"success": False, "result": "Command timed out", "method": method}
        except ValueError as e:
            return {"success": False, "result": f"Invalid input: {e}", "method": method}
        # Unexpected errors propagate — they're bugs, not user errors

    async def execute_workflow(self, steps: list[str], params: dict,
                               ask_user=None) -> list[dict]:
        """Execute a multi-step workflow. Returns list of step results.

        ask_user: async callable(question) → str for getting user input
        """
        results = []
        for step in steps:
            # Parse step — may have suffix like "create_file:README.md"
            if ":" in step and not step.startswith("bash:"):
                intent, hint = step.split(":", 1)
            elif step.startswith("bash:"):
                intent = step
                hint = None
            else:
                intent = step
                hint = None

            # Build step-specific params
            step_params = dict(params)  # copy base params

            if intent == "git_commit" and "message" not in step_params:
                if ask_user:
                    msg = await ask_user("Commit message?")
                    step_params["message"] = msg
                else:
                    step_params["message"] = "auto-commit"

            if intent == "create_file" and hint:
                # e.g., "create_file:README.md"
                step_params.setdefault("path", os.path.join(self.working_dir, hint))
                step_params.setdefault("content", f"# {os.path.basename(hint).replace('.md','')}\n")

            if intent == "create_dir" and "path" not in step_params:
                step_params["path"] = self.working_dir

            if intent == "git_add" and "files" not in step_params:
                step_params["files"] = ["."]

            # Execute
            if intent.startswith("bash:"):
                cmd = intent.split(":", 1)[1]
                result = self._call_bash(cmd, step_params)
                results.append({"step": intent, "success": True, "result": result})
            else:
                result = await self.execute(intent, step_params)
                results.append({"step": intent, **result})

                # Stop workflow if a step fails
                if not result["success"]:
                    break

        return results

    def _resolve(self, path: str, allow_parent: bool = False) -> str:
        """Resolve a path relative to working_dir.

        Contract:
        - REQUIRES: path is a string
        - ENSURES: returned path is absolute and resolved (no symlinks)
        - ENSURES: if allow_parent=False, path must be within working_dir
        - RAISES: TypeError if path is not str
        - RAISES: ValueError if path escapes working_dir (when allow_parent=False)
        """
        if not isinstance(path, str):
            raise TypeError(f"path must be str, got {type(path)}")
        if not path or path == ".":
            return self.working_dir
        if os.path.isabs(path):
            resolved = os.path.realpath(path)
        else:
            resolved = os.path.realpath(os.path.join(self.working_dir, path))
        # Security: file operations must stay within working_dir
        if not allow_parent:
            base = os.path.realpath(self.working_dir)
            if not (resolved.startswith(base + os.sep) or resolved == base):
                raise ValueError(f"Path escapes working directory: {path} → {resolved}")
        return resolved

    # ----- Bash -----

    def _call_bash(self, command: str, params: dict) -> str:
        """Execute a command via subprocess with argument lists (no shell).

        Contract:
        - REQUIRES: command is a known command prefix from INTENT_ROUTES
        - ENSURES: no shell interpretation of user params (args passed as list)
        - ENSURES: subprocess runs with 30s timeout in working_dir
        """
        args = self._build_args(command, params)
        r = subprocess.run(
            args, capture_output=True, text=True,
            timeout=30, cwd=self.working_dir,
        )
        output = r.stdout.strip() or r.stderr.strip() or "Done."
        if r.returncode == 0:
            return output
        if "nothing to commit" in output or "nothing added to commit" in output:
            return output
        return f"Failed: {output}"

    @staticmethod
    def _build_args(command: str, params: dict) -> list[str]:
        """Build subprocess argument list from command + params. No shell."""
        if command == "git status":
            return ["git", "status", "--short"]
        elif command == "git add":
            files = params.get("files", ["."])
            if isinstance(files, str):
                files = [files]
            return ["git", "add"] + files
        elif command == "git commit":
            return ["git", "commit", "-m", params.get("message", "Update")]
        elif command == "git diff":
            return ["git", "diff"]
        elif command.startswith("git log"):
            return ["git", "log", "--oneline", "-20"]
        elif command == "git branch":
            name = params.get("branch_name", "")
            return ["git", "checkout", "-b", name] if name else ["git", "branch", "-a"]
        elif command == "git push":
            args = ["git", "push", params.get("remote", "origin")]
            if params.get("branch"):
                args.append(params["branch"])
            return args
        elif command == "git pull":
            args = ["git", "pull", params.get("remote", "origin")]
            if params.get("branch"):
                args.append(params["branch"])
            return args
        elif command == "git merge":
            return ["git", "merge", params.get("branch_name", "")]
        elif command == "git clone":
            return ["git", "clone", params.get("url", "")]
        elif command == "git init":
            return ["git", "init"]
        elif command == "pytest":
            args = ["python3", "-m", "pytest", "-v"]
            if params.get("path"):
                args.insert(3, params["path"])
            return args
        elif command == "pip install":
            pkg = params.get("package", "")
            return ["pip", "install", pkg] if pkg else ["pip", "install", "-r", "requirements.txt"]
        elif command == "uvicorn":
            port = str(params.get("port", "8000"))
            return ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", port]
        elif command == "black":
            return ["black", params.get("path", ".")]
        elif command == "open":
            return ["open", params.get("url", "http://localhost:8000")]
        else:
            return command.split()  # fallback: split on whitespace

    # ----- Internal (state changes) -----

    def _call_internal(self, func: str, params: dict) -> str:
        if func == "change_dir":
            path = params.get("path", "")
            if not path:
                return f"Current directory: {self.working_dir}"
            # Nav aliases (parent, back, up) are resolved by param_extractor
            new_dir = self._resolve(path, allow_parent=True)
            # Bound: cannot cd above the root (initial working_dir)
            if not (new_dir.startswith(self._root + os.sep) or new_dir == self._root):
                return f"Cannot navigate above {self._root}"
            if os.path.isdir(new_dir):
                self.working_dir = new_dir
                return f"Changed directory to {new_dir}"
            else:
                return f"Directory not found: {path}"
        return f"Unknown internal function: {func}"

    # ----- Local Python -----

    def _call_local(self, func: str, params: dict) -> str:
        if func == "current_dir":
            return self.working_dir

        elif func == "list_files":
            target = self._resolve(params.get("path", "."))
            if not os.path.isdir(target):
                return f"Not a directory: {params.get('path', '.')}"
            entries = sorted(os.listdir(target))
            lines = []
            for name in entries:
                full = os.path.join(target, name)
                prefix = "[DIR]" if os.path.isdir(full) else "[FILE]"
                lines.append(f"{prefix} {name}")
            return "\n".join(lines) if lines else "(empty directory)"

        elif func == "create_file":
            path = params.get("path", "")
            if not path:
                return "Error: path required"
            full = self._resolve(path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            content = params.get("content", "")
            content = content.replace("\\n", "\n").replace("\\t", "\t")
            with open(full, "w") as f:
                f.write(content)
            return f"Successfully wrote to {full}"

        elif func == "read_file":
            path = params.get("path", "")
            if not path:
                return "Error: path required"
            full = self._resolve(path)
            if not os.path.exists(full):
                return f"File not found: {path}"
            with open(full) as f:
                return f.read()

        elif func == "edit_file":
            path = params.get("path", "")
            if not path:
                return "Error: path required"
            full = self._resolve(path)
            if not os.path.exists(full):
                return f"File not found: {path}"
            old = params.get("old", "")
            new = params.get("new", "")
            if not old:
                return "Error: old text required"
            with open(full) as f:
                content = f.read()
            if old not in content:
                return f"Text not found in {path}"
            content = content.replace(old, new, 1)
            with open(full, "w") as f:
                f.write(content)
            return f"Edited {path}"

        elif func == "create_dir":
            path = params.get("path", "")
            if not path:
                return "Error: path required"
            full = self._resolve(path)
            os.makedirs(full, exist_ok=True)
            return f"Successfully created directory {full}"

        elif func == "move_file":
            src = params.get("source", "")
            dst = params.get("destination", "")
            if not src or not dst:
                return "Error: source and destination required"
            shutil.move(self._resolve(src), self._resolve(dst))
            return f"Moved {src} → {dst}"

        elif func == "copy_file":
            src = params.get("source", "")
            dst = params.get("destination", "")
            if not src or not dst:
                return "Error: source and destination required"
            src_full = self._resolve(src)
            dst_full = self._resolve(dst)
            if os.path.isdir(src_full):
                shutil.copytree(src_full, dst_full)
            else:
                os.makedirs(os.path.dirname(dst_full), exist_ok=True)
                shutil.copy2(src_full, dst_full)
            return f"Copied {src} → {dst}"

        elif func == "delete_file":
            path = params.get("path", "")
            if not path:
                return "Error: path required"
            full = self._resolve(path)
            if os.path.isdir(full):
                shutil.rmtree(full)
            elif os.path.exists(full):
                os.remove(full)
            else:
                return f"Not found: {path}"
            return f"Deleted {path}"

        elif func == "search_files":
            pattern = params.get("pattern", "")
            if not pattern:
                return "Error: pattern required"
            import fnmatch
            matches = []
            for root, dirs, files in os.walk(self.working_dir):
                for name in files:
                    if fnmatch.fnmatch(name, pattern) or pattern.lower() in name.lower():
                        rel = os.path.relpath(os.path.join(root, name), self.working_dir)
                        matches.append(rel)
            return "\n".join(matches[:50]) if matches else f"No files matching: {pattern}"

        return f"Unknown local function: {func}"

    async def close(self):
        """No async resources to clean up."""
        pass
