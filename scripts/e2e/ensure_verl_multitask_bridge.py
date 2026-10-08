#!/usr/bin/env python3
"""Ensure the *imported* VERL Fully Async entry has the optional MultiTask bridge.

One-command setup for real E2E: try the canonical patch on an unchanged VERL
checkout, otherwise apply the SAME bridge contract with narrow AST-guided edits.
Never switch which VERL installation Python imports; never reset user changes.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from importlib.util import find_spec
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


class BridgeSetupError(RuntimeError):
    """An unsafe or unrecognized VERL entry; no blind text replacement."""


def _added_block(patch: str, first_line: str) -> str:
    """Reuse the bridge source/defaults declared in the checked-in .patch."""
    lines = patch.splitlines()
    start = next((i for i, line in enumerate(lines) if line == "+" + first_line), None)
    if start is None:
        raise BridgeSetupError(f"bridge patch is missing {first_line!r}")
    result = []
    for line in lines[start:]:
        if not line.startswith("+") or line.startswith("+++"):
            break
        result.append(line[1:])
    return "\n".join(result).rstrip() + "\n"


def _main_function(tree: ast.Module) -> ast.FunctionDef:
    candidates = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    if len(candidates) != 1 or not candidates[0].decorator_list:
        raise BridgeSetupError("expected exactly one decorated VERL Fully Async main(config)")
    return candidates[0]


def _edit_entry(source: str, patch_text: str) -> str:
    tree = ast.parse(source)
    main = _main_function(tree)
    resolvers = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_resolve_task_runner_class"
    ]
    if len(resolvers) > 1:
        raise BridgeSetupError("duplicate MultiTask resolver functions")
    calls = [
        n for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "run_ppo"
    ]
    if len(calls) != 1:
        raise BridgeSetupError("expected exactly one run_ppo call in Fully Async main")
    runner_kwargs = [kw for kw in calls[0].keywords if kw.arg == "task_runner_class"]
    if len(runner_kwargs) != 1:
        raise BridgeSetupError("run_ppo must have one explicit task_runner_class argument")
    value = runner_kwargs[0].value
    already_connected = (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "_resolve_task_runner_class"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "config"
        and not value.keywords
    )
    # The canonical patch instead assigns task_runner_class locally.
    if isinstance(value, ast.Name) and value.id == "task_runner_class":
        already_connected = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "task_runner_class" for t in n.targets)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "_resolve_task_runner_class"
            for n in ast.walk(main)
        )
    native = isinstance(value, ast.Name) and value.id == "FullyAsyncTaskRunner"
    if not native and not already_connected:
        raise BridgeSetupError(
            "custom run_ppo task_runner_class detected; refusing to overwrite its selection"
        )

    lines = source.splitlines(keepends=True)
    if native:
        if value.lineno != value.end_lineno:
            raise BridgeSetupError("unexpected multiline native TaskRunner expression")
        line = lines[value.lineno - 1].encode("utf-8")
        start, end = value.col_offset, value.end_col_offset
        if line[start:end] != b"FullyAsyncTaskRunner":
            raise BridgeSetupError("AST/source mismatch at TaskRunner selection")
        lines[value.lineno - 1] = (
            line[:start] + b"_resolve_task_runner_class(config)" + line[end:]
        ).decode("utf-8")
    if not resolvers:
        helper = _added_block(patch_text, "def _resolve_task_runner_class(config):")
        ast.parse(helper)
        first_decorator = min(n.lineno for n in main.decorator_list)
        lines.insert(first_decorator - 1, helper + "\n")
    modified = "".join(lines)
    ast.parse(modified)
    return modified


def _edit_config(source: str, patch_text: str) -> str:
    existing = re.search(r"(?m)^multitask:\s*(?:#.*)?$", source)
    if existing:
        # Never change a preexisting MultiTask policy/configuration.
        section = source[existing.end():]
        stop = re.search(r"(?m)^[A-Za-z_][\w-]*\s*:", section)
        section = section[:stop.start()] if stop else section
        if not (re.search(r"(?m)^  enabled\s*:", section)
                and re.search(r"(?m)^  runtime\s*:", section)
                and re.search(r"(?m)^    profile\s*:", section)):
            raise BridgeSetupError(
                "existing multitask YAML block lacks enabled/runtime.profile; preserve and inspect manually"
            )
        return source
    matches = list(re.finditer(r"(?m)^async_training\s*:", source))
    if len(matches) != 1:
        raise BridgeSetupError("cannot uniquely locate async_training in VERL Hydra YAML")
    defaults = _added_block(patch_text, "multitask:")
    return source[:matches[0].start()] + defaults + "\n" + source[matches[0].start():]


def _files_for_entry(entry_path: Path) -> tuple[Path, Path, Path]:
    entry_path = entry_path.resolve(strict=True)
    if entry_path.name != "fully_async_main.py" or entry_path.parent.name != "fully_async_policy":
        raise BridgeSetupError(f"unexpected VERL entry location: {entry_path}")
    root = entry_path.parents[3]
    yaml_path = entry_path.parent / "config" / "fully_async_ppo_trainer.yaml"
    if not yaml_path.is_file():
        raise BridgeSetupError(f"missing VERL Fully Async config: {yaml_path}")
    if any(part in ("site-packages", "dist-packages") for part in entry_path.parts):
        raise BridgeSetupError(
            "imported VERL is inside site-packages; install an editable source checkout "
            "rather than modifying installed package files in place: " + str(entry_path)
        )
    return root, entry_path, yaml_path


def ensure_current_verl_bridge(*, entry_path: Path | None = None,
                               patch_path: Path | None = None,
                               check_only: bool = False) -> bool:
    """Return True if files changed; safe to call repeatedly.

    entry_path/patch_path overrides exist for offline unit tests; normal E2E
    always resolves the active interpreter's VERL module automatically.
    """
    if entry_path is None:
        spec = find_spec("verl.experimental.fully_async_policy.fully_async_main")
        if spec is None or not spec.origin:
            raise BridgeSetupError(f"VERL Fully Async entry not importable by {sys.executable}")
        entry_path = Path(spec.origin)
    if patch_path is None:
        patch_path = Path(__file__).resolve().parents[2] / "patches" / "verl-v0.10-fully-async-multitask-entry.patch"
    root, entry, yaml = _files_for_entry(Path(entry_path))
    patch_text = Path(patch_path).read_text(encoding="utf-8")
    before = {entry: entry.read_text(encoding="utf-8"),
              yaml: yaml.read_text(encoding="utf-8")}
    after = {entry: _edit_entry(before[entry], patch_text),
             yaml: _edit_config(before[yaml], patch_text)}
    changed = [path for path in before if before[path] != after[path]]
    print(f"[MT-BRIDGE] python={sys.executable} imported_VERL={entry}", flush=True)
    if not changed:
        print("[MT-BRIDGE] READY: MultiTask entry and Hydra config already connected", flush=True)
        return False
    if check_only:
        raise BridgeSetupError(
            "MultiTask bridge is not installed in the imported VERL; "
            "rerun without --check-only"
        )
    if any(not os.access(path, os.W_OK) for path in changed):
        raise BridgeSetupError("VERL source is not writable: " + ", ".join(map(str, changed)))

    # Back up every touched file BEFORE git apply or surgical fallback.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backups = {}
    for path in changed:
        backup = path.with_name(path.name + f".mtbridge-{stamp}.bak")
        shutil.copy2(path, backup)
        backups[path] = backup
    command = ["git", "-C", str(root), "apply"]
    patch = str(Path(patch_path).resolve())
    can_apply = (
        len(changed) == 2
        and (root / ".git").exists()
        and subprocess.run(command + ["--check", patch], capture_output=True).returncode == 0
    )
    try:
        if can_apply:
            subprocess.run(command + [patch], check=True, capture_output=True, text=True)
            method = "repository patch"
        else:
            for path in changed:
                path.write_text(after[path], encoding="utf-8")
            method = "compatible AST-guided patch"
        current_entry = entry.read_text(encoding="utf-8")
        current_yaml = yaml.read_text(encoding="utf-8")
        # Idempotent static verification, also detects unexpected patch side effects.
        if _edit_entry(current_entry, patch_text) != current_entry:
            raise BridgeSetupError("post-apply VERL entry verification failed")
        if _edit_config(current_yaml, patch_text) != current_yaml:
            raise BridgeSetupError("post-apply VERL Hydra config verification failed")
    except BaseException:
        for path, backup in backups.items():
            shutil.copy2(backup, path)
        raise
    print(f"[MT-BRIDGE] APPLIED: {method}", flush=True)
    for backup in backups.values():
        print(f"[MT-BRIDGE] backup={backup}", flush=True)
    print("[MT-BRIDGE] READY: imported VERL now selects MultiTask when enabled", flush=True)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only", action="store_true",
                   help="verify without writing the VERL checkout")
    args = p.parse_args()
    try:
        ensure_current_verl_bridge(check_only=args.check_only)
    except (BridgeSetupError, SyntaxError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[MT-BRIDGE] BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
