#!/usr/bin/env python3
"""SACH loop guard for Claude Code PreToolUse, PostToolUse, PostToolUseFailure.

Blocks the third identical failed attempt in an unchanged workspace, or a failed
A/B/A followed by B. Successful polling and diagnostic reads remain available.
This is loop detection, not a shell sandbox or an authorization mechanism.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


READ_TOOLS = {"Read", "Glob", "Grep", "WebFetch", "WebSearch", "ListMcpResourcesTool", "ReadMcpResourceTool"}
EVENTS = {"PreToolUse", "PostToolUse", "PostToolUseFailure"}


def compute_action_fingerprint(tool_name, params):
    # Presentation-only description is excluded. Preserve all operational values,
    # including float precision, whitespace inside strings, and list order.
    params = {k: v for k, v in params.items() if k != "description"}
    raw = json.dumps([tool_name, params], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def workspace_signature(cwd):
    """Cheap change discriminator, not a cryptographic proof of workspace contents."""
    try:
        opts = dict(cwd=cwd, capture_output=True, timeout=3, check=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], **opts).stdout.decode().strip()
        raw = subprocess.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], **opts).stdout
        fields, records, i = raw.split(b"\0"), [], 0
        while i < len(fields):
            entry = fields[i]; i += 1
            if len(entry) < 4: continue
            names = [entry[3:]]
            if b"R" in entry[:2] or b"C" in entry[:2]:
                names.append(fields[i]); i += 1
            for name in names:
                relative = os.fsdecode(name)
                if relative.startswith((".agents/state/", ".agents/logs/", ".agent/state/", ".agent/logs/")) or relative in (".agents/TASK.md", ".agents/task.json", ".agent/TASK.md", ".agent/task.json"):
                    continue
                try:
                    stat = (Path(cwd) / relative).lstat()
                    records.append((relative, entry[:2].decode(), stat.st_size, stat.st_mtime_ns))
                except FileNotFoundError: records.append((relative, "deleted"))
        payload = json.dumps([head, sorted(records)], sort_keys=True)
    except (OSError, subprocess.SubprocessError, IndexError):
        # Without a change discriminator, identical failures may follow valid
        # source edits. Observe outcomes but do not manufacture a no-progress claim.
        return None
    return hashlib.sha256(payload.encode()).hexdigest()


@contextmanager
def state_lock(path, timeout=5):
    """OS-released lock; an interrupted process does not leave a stale lock lease."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a+b") as handle:
        handle.seek(0)
        if not handle.read(1): handle.write(b"0"); handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline: raise ValueError("guard state lock timeout")
                time.sleep(0.02)
        try: yield
        finally:
            if os.name == "nt":
                import msvcrt
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_state(path):
    if not path.exists(): return {"version": 1, "history": [], "blocked": 0}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("history"), list):
        raise ValueError("invalid guard state; inspect and reset this session state explicitly")
    for record in state["history"]:
        if (not isinstance(record, dict) or not isinstance(record.get("fingerprint"), str)
                or not isinstance(record.get("workspace"), (str, type(None))) or not isinstance(record.get("failed"), bool)):
            raise ValueError("invalid guard history")
    return state


def save_state(path, state):
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".guard-", suffix=".tmp", delete=False, encoding="utf-8") as out:
            temp = out.name
            json.dump(state, out, sort_keys=True); out.flush(); os.fsync(out.fileno())
        os.replace(temp, path); temp = None
    finally:
        if temp and os.path.exists(temp): os.unlink(temp)


def process_event(payload, state_path, window_size=10, threshold=3):
    if not 2 <= threshold <= window_size or window_size > 100: raise ValueError("invalid guard thresholds")
    event = payload.get("hook_event_name", "PreToolUse")
    if event not in EVENTS: raise ValueError("unsupported hook event")
    tool = payload.get("tool_name", payload.get("tool"))
    params = payload.get("tool_input", payload.get("params", {}))
    if not isinstance(tool, str) or not tool or not isinstance(params, dict): raise ValueError("tool_name and object tool_input required")
    fingerprint = compute_action_fingerprint(tool, params)
    workspace = payload.get("progress_key") or workspace_signature(payload.get("cwd", os.getcwd()))
    with state_lock(state_path):
        state = load_state(state_path)
        history = state["history"][-window_size:]
        decision, reason = "allow", "No repeated failed trajectory."
        if event == "PreToolUse":
            relevant = [r for r in history if r["workspace"] == workspace]
            # A successful completion of this action invalidates older failures.
            last_success = max((i for i, r in enumerate(relevant) if not r["failed"] and r["fingerprint"] == fingerprint), default=-1)
            failures = [r for r in relevant[last_success + 1:] if r["failed"] and r["fingerprint"] == fingerprint]
            tail = history[-3:]
            oscillation = len(tail) == 3 and all(r["failed"] and r["workspace"] == workspace for r in tail) and tail[0]["fingerprint"] == tail[2]["fingerprint"] and tail[1]["fingerprint"] == fingerprint and tail[0]["fingerprint"] != fingerprint
            if workspace is not None and tool not in READ_TOOLS and (len(failures) >= threshold - 1 or oscillation):
                decision = "deny"
                reason = "Repeated failed action in unchanged workspace. Inspect evidence, change the hypothesis, or replan; diagnostic tools remain available."
                state["blocked"] = state.get("blocked", 0) + 1
        else:
            response = payload.get("tool_response", {})
            failed = event == "PostToolUseFailure" or payload.get("is_error") is True
            if isinstance(response, dict):
                failed = failed or response.get("is_error") is True or any(isinstance(response.get(k), int) and not isinstance(response.get(k), bool) and response[k] != 0 for k in ("exit_code", "exitCode"))
            observation = payload.get("error", response)
            record = {"fingerprint": fingerprint, "workspace": workspace, "failed": bool(failed),
                      "observation_sha256": hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()}
            record["tool_use_id"] = payload.get("tool_use_id")
            if not record["tool_use_id"] or not any(r.get("tool_use_id") == record["tool_use_id"] for r in history):
                history.append(record)
            state["history"] = history[-window_size:]
        save_state(state_path, state)
    if event == "PreToolUse" and decision == "deny":
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                     "permissionDecisionReason": reason}}
    # No opinion on authorization. Let the host's normal permission flow run.
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file")
    parser.add_argument("--tool")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--reset", action="store_true", help="Reset an explicitly selected session state")
    args = parser.parse_args()
    event = "PreToolUse"
    try:
        payload = {} if sys.stdin.isatty() else json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict): raise ValueError("hook payload must be an object")
        event = payload.get("hook_event_name", event)
        if args.tool: payload.update(tool_name=args.tool, tool_input=json.loads(args.params))
        scope = json.dumps([payload.get("session_id", "standalone"), str(Path(payload.get("cwd", os.getcwd())).resolve())])
        key = hashlib.sha256(scope.encode()).hexdigest()[:24]
        state_path = Path(args.state_file) if args.state_file else Path(payload.get("cwd", os.getcwd())) / ".agents/state" / f"guard-{key}.json"
        if args.reset:
            if not args.state_file: raise ValueError("--reset requires --state-file")
            with state_lock(state_path): save_state(state_path, {"version": 1, "history": [], "blocked": 0})
            print(json.dumps({"status": "RESET"})); return 0
        print(json.dumps(process_event(payload, state_path, args.window, args.threshold))); return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # PreToolUse exits 2 to deny on corrupt input/state/persistence failure.
        # Post events cannot undo an already executed tool; expose the diagnostic.
        print(f"SACH guard error: {exc}", file=sys.stderr)
        return 1 if event in ("PostToolUse", "PostToolUseFailure") else 2


if __name__ == "__main__":
    sys.exit(main())
