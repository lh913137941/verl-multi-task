"""Static acceptance-runner validation for CI without Ray/VERL."""

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "scripts" / "e2e"


def test_e2e_shell_scripts_parse_with_bash():
    scripts = sorted(E2E.glob("*.sh"))
    assert scripts
    for path in scripts:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_e2e_python_drivers_parse_without_importing_runtime_dependencies():
    drivers = sorted(E2E.glob("*.py"))
    assert drivers
    for path in drivers:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
