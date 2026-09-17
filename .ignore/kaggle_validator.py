#!/usr/bin/env python3
"""
Kaggle GPU validator for the Paper-to-Code pipeline.
Uploads solution.ipynb as a private Kaggle kernel with GPU enabled,
waits for execution to complete, downloads the log, and checks for errors.
"""

import json, os, re, subprocess, sys, time
from pathlib import Path
from typing import Optional, Tuple, List

REPO_PATH = Path("/Users/nadymini2/labs/research-notebooks")


def set_kaggle_env():
    """Ensure kaggle CLI is on PATH."""
    kaggle_bin = Path.home() / "Library" / "Python" / "3.9" / "bin"
    os.environ["PATH"] = f"{kaggle_bin}:{os.environ.get('PATH', '')}"


def run_kaggle(args, timeout=60):
    set_kaggle_env()
    result = subprocess.run(
        ["kaggle"] + args, capture_output=True, text=True, timeout=timeout
    )
    return result


def make_kernel_metadata(folder: Path, title: str, notebook_file: str = "solution.ipynb"):
    """Create kernel-metadata.json required by kaggle kernels push."""
    # Kaggle slugs: lowercase, alphanumeric + hyphen, max 50 chars
    slug = re.sub(r"[^a-z0-9-]", "", title.lower().replace(" ", "-").replace("_", "-"))
    slug = slug.strip("-")[:50]
    kernel_id = f"nadymsazad/{slug}"
    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": notebook_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    return meta, kernel_id


def inject_notebook_copy_cell(folder: Path):
    """Add a final cell to solution.ipynb that copies the executed notebook
    to /kaggle/working/ so it can be downloaded via kaggle kernels output."""
    nb_path = folder / "solution.ipynb"
    nb = json.loads(nb_path.read_text())

    # Check if cell already exists
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and "__notebook__.ipynb" in "".join(cell.get("source", [])):
            return  # Already injected

    copy_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Auto-injected by kaggle_validator: save executed notebook with outputs ---\n",
            "import shutil, os\n",
            "# Kaggle internally stores the executed notebook as __notebook__.ipynb\n",
            "if os.path.exists('/kaggle/working/__notebook__.ipynb'):\n",
            "    shutil.copy('/kaggle/working/__notebook__.ipynb', '/kaggle/working/executed_solution.ipynb')\n",
            "    print('Executed notebook saved to output folder.')\n",
            "else:\n",
            "    # Fallback: try alternative path\n",
            "    import glob\n",
            "    candidates = glob.glob('/kaggle/working/*.ipynb')\n",
            "    print(f'Available notebooks in output: {candidates}')\n"
        ]
    }
    nb["cells"].append(copy_cell)
    nb_path.write_text(json.dumps(nb, indent=1))
    print("Injected __notebook__.ipynb copy cell into solution.ipynb")


def push_kernel(folder: Path, title: str) -> str:
    """Push the notebook folder as a Kaggle kernel. Returns kernel_id."""
    nb_path = folder / "solution.ipynb"
    if not nb_path.exists():
        raise FileNotFoundError(f"solution.ipynb not found in {folder}")

    # Inject the notebook-copy cell before pushing
    inject_notebook_copy_cell(folder)

    # Kaggle expects kernel-metadata.json in the same dir as the notebook
    meta, kernel_id = make_kernel_metadata(folder, title)
    meta_path = folder / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    result = run_kaggle(["kernels", "push", "-p", str(folder)], timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"kaggle kernels push failed: {result.stdout} {result.stderr}")
    print(result.stdout.strip())
    return kernel_id


def wait_for_kernel(kernel_id: str, max_wait_seconds: int = 5400, poll_interval: int = 20) -> str:
    """Poll Kaggle kernel status until complete or timeout. Returns final status."""
    print(f"Waiting for {kernel_id} to finish (max {max_wait_seconds}s)...")
    elapsed = 0
    while elapsed < max_wait_seconds:
        result = run_kaggle(["kernels", "status", kernel_id], timeout=60)
        status_line = result.stdout.strip()
        print(f"  [{elapsed:4d}s] {status_line}")
        if result.returncode == 0 and "COMPLETE" in status_line.upper():
            return "COMPLETE"
        if result.returncode == 0 and any(x in status_line.upper() for x in ["ERROR", "FAILED", "CANCEL"]):
            return status_line
        time.sleep(poll_interval)
        elapsed += poll_interval
    return "TIMEOUT"


def download_log(kernel_id: str, output_dir: Path) -> Path:
    """Download kernel log to output_dir. Returns log file path."""
    result = run_kaggle(["kernels", "output", kernel_id, "-p", str(output_dir)], timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"kaggle kernels output failed: {result.stdout} {result.stderr}")
    print(result.stdout.strip())
    # log file name is based on kernel slug
    slug = kernel_id.split("/")[-1]
    log_file = output_dir / f"{slug}.log"
    if log_file.exists():
        return log_file
    # fallback: any .log file
    logs = list(output_dir.glob("*.log"))
    if logs:
        return logs[0]
    raise FileNotFoundError(f"No log file found in {output_dir}")


def check_log_for_errors(log_path: Path) -> List[str]:
    """Parse Kaggle log JSON lines and return list of error messages."""
    issues = []
    content = log_path.read_text()
    # Log is a JSON array of stream events
    try:
        events = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: plain text scan
        if "error" in content.lower():
            issues.append("Log contains 'error' (plain text scan)")
        return issues

    for ev in events:
        stream = ev.get("stream_name", "")
        data = ev.get("data", "")
        # Kernel-level error indicators
        if stream == "stderr" and not any(w in data.lower() for w in ["warning", "debugger", "frozen modules", "pydevd"]):
            # Genuine stderr that is not a common warning
            if "error" in data.lower() or "exception" in data.lower() or "traceback" in data.lower():
                issues.append(f"stderr: {data[:200]}")
    return issues


def download_executed_notebook(kernel_id: str, output_dir: Path, solution_path: Path) -> bool:
    """Download executed_solution.ipynb from Kaggle output and replace local solution.ipynb.
    Returns True if replacement succeeded."""
    executed_path = output_dir / "executed_solution.ipynb"
    if executed_path.exists():
        nb = json.loads(executed_path.read_text())
        solution_path.write_text(json.dumps(nb, indent=1))
        print(f"Replaced {solution_path.name} with executed version ({executed_path.stat().st_size} bytes)")
        return True
    print("executed_solution.ipynb not found in Kaggle output — keeping source-only notebook")
    return False


def make_kernel_public(folder: Path, title: str) -> bool:
    """Re-push the kernel with is_private=false to make it public on Kaggle.
    Call this only after validation has passed. Returns True if push succeeded."""
    meta, kernel_id = make_kernel_metadata(folder, title)
    meta["is_private"] = "false"
    meta_path = folder / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    result = run_kaggle(["kernels", "push", "-p", str(folder)], timeout=120)
    if result.returncode != 0:
        print(f"WARNING: Failed to make kernel public: {result.stdout} {result.stderr}")
        return False
    print(f"Kernel {kernel_id} is now PUBLIC on Kaggle.")
    return True


def validate(folder: Path, title: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Run Kaggle GPU validation. Returns (ok, issues)."""
    try:
        if title is None:
            # Derive title from folder name
            title = folder.name.replace("-", " ").replace("_", " ").title()

        kernel_id = push_kernel(folder, title)
        status = wait_for_kernel(kernel_id)

        if status != "COMPLETE":
            return False, [f"Kaggle kernel did not complete: {status}"]

        output_dir = folder / ".kaggle_output"
        output_dir.mkdir(exist_ok=True)
        log_path = download_log(kernel_id, output_dir)
        print(f"Log downloaded: {log_path}")

        issues = check_log_for_errors(log_path)

        # Replace local solution.ipynb with executed version (outputs included)
        solution_path = folder / "solution.ipynb"
        download_executed_notebook(kernel_id, output_dir, solution_path)

        # If validation passed, make the kernel public on Kaggle
        if len(issues) == 0:
            make_kernel_public(folder, title)

        return len(issues) == 0, issues

    except Exception as e:
        return False, [f"Kaggle validation exception: {e}"]


def main():
    import argparse, shutil
    parser = argparse.ArgumentParser(description="Validate solution.ipynb on Kaggle GPU")
    parser.add_argument("folder", help="Folder containing solution.ipynb")
    parser.add_argument("--title", help="Kaggle kernel title (optional)")
    args = parser.parse_args()

    folder = Path(args.folder)
    ok, issues = validate(folder, title=args.title)

    for issue in issues:
        print(f"[VALIDATION ISSUE] {issue}")

    if ok:
        print("[VALIDATION OK] solution.ipynb executed successfully on Kaggle GPU.")
        return 0
    else:
        print("[VALIDATION FAILED] solution.ipynb has errors on Kaggle GPU.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
