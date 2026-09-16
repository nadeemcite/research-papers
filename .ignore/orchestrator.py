#!/usr/bin/env python3
"""
Paper-to-Code Notebook Builder — Orchestrator
Reads Google Sheet, picks first 'Not started' paper, builds the 4-file package,
commits + pushes to GitHub, and updates sheet status.
"""

import json, os, re, sys, subprocess, time, traceback
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

# ─── Config ───────────────────────────────────────────────────────────────────
REPO_PATH = Path("/Users/nadymini2/labs/research-notebooks")
CRED_FILE = REPO_PATH / ".ignore" / "gcscredentials.json"
SHEET_ID = "1EzTm-kSGP1Y1GIW7nqKD6-5moKDmAPHPCAJZitdSFVg"
WORKSHEET_GID = 1429419384
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

STATUS_COL = 5  # Column E (1-indexed)
SR_COL = 1     # Column A


# ─── Google Sheet helpers ─────────────────────────────────────────────────────
def get_sheet():
    creds = Credentials.from_service_account_file(str(CRED_FILE), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    for ws in sh.worksheets():
        if str(ws.id) == str(WORKSHEET_GID):
            return ws
    return sh.sheet1


def get_next_row(ws):
    """Return (row_index_1based, row_data_dict) for first 'Not started' row, or None."""
    all_data = ws.get_all_values()
    headers = all_data[0]
    for i, row in enumerate(all_data[1:], start=2):  # row 2 = first data row in sheet
        row_dict = dict(zip(headers, row))
        if row_dict.get("Status", "").strip() == "Not started":
            return i, row_dict
    return None


def set_status(ws, sheet_row, status):
    ws.update_cell(sheet_row, STATUS_COL, status)


def set_note(ws, sheet_row, note):
    """Try to write to a Notes column (col F=6) if it exists, else append."""
    try:
        ws.update_cell(sheet_row, 6, note)
    except Exception:
        pass  # No Notes column — silently skip


# ─── Folder naming ────────────────────────────────────────────────────────────
def make_slug(title):
    slug = re.sub(r"[^a-zA-Z0-9\s-]", "", title.lower())
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    if len(slug) > 40:
        slug = slug[:40]
        slug = slug[:slug.rfind("-")] if "-" in slug else slug[:40]
    return slug


def make_folder_name(sr, title):
    return f"{int(sr):03d}_{make_slug(title)}"


# ─── Git helpers ──────────────────────────────────────────────────────────────
def git(args, cwd=REPO_PATH):
    result = subprocess.run(
        ["git"] + args, cwd=str(cwd), capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def load_ssh_keys():
    """Ensure both GitHub deploy keys are loaded in ssh-agent."""
    keys = ["~/.ssh/nadyth.ssh", "~/.ssh/nadeem.ssh"]
    for key in keys:
        key_path = os.path.expanduser(key)
        if os.path.exists(key_path):
            # ssh-add is idempotent; ignore AlreadyInAgent errors
            subprocess.run(["ssh-add", key_path], capture_output=True)


def git_commit_push(folder_path, sr_padded, paper_title):
    load_ssh_keys()
    folder_rel = folder_path.relative_to(REPO_PATH)
    # Stage the new folder
    git(["add", str(folder_rel / "README.md"), str(folder_rel / "solution.ipynb"),
         str(folder_rel / "CODE_ARCHITECTURE.md"), str(folder_rel / "TALK.md")])
    # Also update root README.md if it exists
    root_readme = REPO_PATH / "README.md"
    if root_readme.exists():
        git(["add", "README.md"])
    commit_msg = f"Add: {sr_padded} {paper_title}"
    git(["commit", "-m", commit_msg])
    # Push to both remotes; if one fails, the whole operation fails and sheet reverts
    git(["push", "origin", "main"])
    git(["push", "backup", "main"])


# ─── Main trigger logic ───────────────────────────────────────────────────────
def run_cycle():
    log("=== Cycle start ===")
    ws = get_sheet()

    picked = get_next_row(ws)
    if not picked:
        log("No 'Not started' rows found. Nothing to do this cycle.")
        return

    sheet_row, row_data = picked
    sr = row_data["Sr"]
    title = row_data["Paper Title"]
    arxiv_link = row_data["arXiv Link"]
    code_template = row_data["Code Template / What the Notebook Should Build"]

    log(f"Picked: Sr={sr}, Title={title}")

    # Mark In progress immediately
    set_status(ws, sheet_row, "In progress")
    log(f"Status → In progress (sheet row {sheet_row})")

    folder_name = make_folder_name(sr, title)
    folder_path = REPO_PATH / folder_name
    sr_padded = f"{int(sr):03d}"

    try:
        folder_path.mkdir(parents=True, exist_ok=False)
        log(f"Created folder: {folder_name}")

        # Write a spec file for the notebook builder
        spec = {
            "sr": sr,
            "sr_padded": sr_padded,
            "title": title,
            "arxiv_link": arxiv_link,
            "code_template": code_template,
            "folder": str(folder_path),
        }
        spec_file = folder_path / ".notebook_spec.json"
        spec_file.write_text(json.dumps(spec, indent=2))

        validate_notebook(folder_path)

        # Commit and push the complete package
        git_commit_push(folder_path, sr_padded, title)

        # Mark done only after successful push
        set_status(ws, sheet_row, "Done")
        log(f"Status → Done (sheet row {sheet_row})")

    except Exception as e:
        log(f"FAILURE: {e}")
        traceback.print_exc()
        # Revert status
        set_status(ws, sheet_row, "Not started")
        set_note(ws, sheet_row, f"Failed: {str(e)[:200]}")
        log(f"Status → Not started (reverted)")
        # Clean up partial folder
        if folder_path.exists():
            import shutil
            shutil.rmtree(folder_path, ignore_errors=True)
        raise


def validate_notebook(folder_path: Path) -> None:
    """Run the standalone validator on the generated notebook."""
    validator = REPO_PATH / ".ignore" / "validator.py"
    log(f"Running validator on {folder_path}")
    result = subprocess.run(
        ["python3", str(validator), str(folder_path)],
        cwd=str(REPO_PATH),
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        raise RuntimeError("Notebook validation failed — see issues above. Commit/push blocked.")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


if __name__ == "__main__":
    run_cycle()