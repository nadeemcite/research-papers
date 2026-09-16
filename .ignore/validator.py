#!/usr/bin/env python3
"""
Notebook validator for the Paper-to-Code pipeline.
Runs after solution.ipynb is generated and before commit/push.
Exits with code 0 only if the notebook is fully executed and error-free.
"""

import json, re, subprocess, sys, tempfile
from pathlib import Path


def find_solution_notebook(folder: Path) -> Path:
    nb = folder / "solution.ipynb"
    if not nb.exists():
        raise FileNotFoundError(f"solution.ipynb not found in {folder}")
    return nb


def check_cell_sources(nb: dict) -> list[str]:
    """Static checks on source code for common authoring mistakes."""
    issues = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        # 1. Literal newlines inside f-string quotes (causes SyntaxError)
        # Heuristic: detect f" or f' followed by a newline before the closing quote.
        for match in re.finditer(r'(f["\'])', src):
            start = match.start()
            quote = src[start + 1]
            # find the closing quote not preceded by backslash
            end = None
            escaped = False
            for j in range(start + 2, len(src)):
                ch = src[j]
                if ch == "\\":
                    escaped = not escaped
                    continue
                if ch == quote and not escaped:
                    end = j
                    break
                escaped = False
            if end is not None:
                segment = src[start:end + 1]
                if "\n" in segment:
                    issues.append(
                        f"Cell {i}: f-string contains a literal newline at line break "
                        f"(use '\\n' escape or split into multiple print calls)"
                    )

        # 2. Basic syntax check by compiling the cell
        try:
            compile(src, f"<cell-{i}>", "exec")
        except SyntaxError as e:
            issues.append(f"Cell {i}: SyntaxError — {e.msg} (line {e.lineno})")
        except Exception as e:
            issues.append(f"Cell {i}: Compile error — {type(e).__name__}: {e}")

    return issues


def check_no_error_outputs(nb: dict) -> list[str]:
    """Ensure no cell has an 'error' output type."""
    issues = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "")
                issues.append(f"Cell {i}: stored error output — {ename}: {evalue[:200]}")
    return issues


def execute_notebook(nb_path: Path, timeout: int = 900) -> tuple[bool, list[str]]:
    """Execute the notebook with jupyter nbconvert and return (ok, issues).
    Default timeout raised to 900s (15 min) because training-heavy notebooks
    (PyTorch on MNIST/Fashion-MNIST) can take several minutes to finish.
    """
    issues = []
    nb_path = nb_path.resolve()
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(nb_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(nb_path.parent)
        )
    except subprocess.TimeoutExpired:
        return False, [f"Notebook execution timed out after {timeout}s"]
    except FileNotFoundError:
        return False, ["jupyter nbconvert not found; install jupyter to validate"]

    if result.returncode != 0:
        issues.append(f"jupyter nbconvert failed (exit {result.returncode}):")
        issues.append(result.stderr[-2000:] or result.stdout[-2000:])
        return False, issues

    # Re-read the executed notebook to confirm no errors were produced
    nb = json.loads(nb_path.read_text())
    exec_errors = check_no_error_outputs(nb)
    if exec_errors:
        return False, exec_errors

    return True, []


def validate(folder: Path, execute: bool = True) -> tuple[bool, list[str]]:
    """Run all validations. Returns (ok, list_of_issues)."""
    try:
        nb_path = find_solution_notebook(folder)
        nb = json.loads(nb_path.read_text())
    except Exception as e:
        return False, [f"Could not load notebook: {e}"]

    issues = []
    issues.extend(check_cell_sources(nb))
    issues.extend(check_no_error_outputs(nb))

    if execute:
        ok, exec_issues = execute_notebook(nb_path)
        issues.extend(exec_issues)
        return len(issues) == 0, issues

    return len(issues) == 0, issues


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate a generated solution.ipynb")
    parser.add_argument("folder", help="Folder containing solution.ipynb")
    parser.add_argument(
        "--no-execute", action="store_true",
        help="Skip full execution with jupyter nbconvert"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    ok, issues = validate(folder, execute=not args.no_execute)

    for issue in issues:
        print(f"[VALIDATION ISSUE] {issue}")

    if ok:
        print("[VALIDATION OK] solution.ipynb is clean and error-free.")
        return 0
    else:
        print("[VALIDATION FAILED] solution.ipynb has errors; commit/push blocked.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
