#!/usr/bin/env python3
"""
kata_orchestrator.py

Production-oriented CLI orchestrator for executing untrusted connector code inside
Kata Containers via Docker runtime `kata-runtime`.

Core contract:
- Accepts and forwards top-level connector nodes as-is (including dynamic structures)
- Writes `files` map to /run/code preserving directories
- Validates top-level `requirements` entries and writes requirements.txt
- Resolves {{env:VAR_NAME}} placeholders recursively in connector strings
- Merges input.configuration shallowly into connector.configuration before validation/forwarding
- Validates merged configuration using JSON Schema generated from configurationTypes
- Runs code in container with strict runtime controls and seccomp profile
- Prints exactly one JSON object to stdout as final result

Prerequisites:
- Docker installed and reachable in PATH
- Kata runtime available as `kata-runtime`
- Image available/pullable: `python:3.12-slim`

Example:
  python kata_orchestrator.py connector.json --input input.json --strict
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError
except Exception:
    Draft202012Validator = None  # type: ignore[assignment]
    ValidationError = Exception  # type: ignore[assignment]


RE_REQUIREMENT = re.compile(r"^[A-Za-z0-9._-]+(==[A-Za-z0-9._-]+)?$")
RE_PLACEHOLDER = re.compile(r"\{\{env:([A-Z0-9_]+)\}\}")
MAX_LOG_BYTES = 1024 * 1024  # 1MB
DEFAULT_TIMEOUT_MS = 60000
DEFAULT_MEMORY_MB = 256
DEFAULT_CPUS = 0.5
DEFAULT_PIDS_LIMIT = 8
DEFAULT_ENTRYPOINT = "runner_entrypoint.py"
DEFAULT_IMAGE = "python:3.12-slim"


class OrchestratorError(Exception):
    """Domain error for orchestrator failures."""


class TimeoutError(OrchestratorError):
    """Raised on sandbox wall-clock timeout."""


class SecurityScanResult:
    def __init__(self) -> None:
        self.findings: list[str] = []

    def add(self, message: str) -> None:
        self.findings.append(message)

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)


class SuspiciousPythonVisitor(ast.NodeVisitor):
    """AST visitor to detect suspicious Python usage patterns."""

    def __init__(self, file_path: Path, result: SecurityScanResult) -> None:
        self.file_path = file_path
        self.result = result

    def _record(self, node: ast.AST, detail: str) -> None:
        lineno = getattr(node, "lineno", "?")
        self.result.add(f"{self.file_path}:{lineno}: {detail}")

    def visit_Import(self, node: ast.Import) -> Any:
        suspicious_imports = {"ctypes", "cffi", "subprocess"}
        for alias in node.names:
            mod = alias.name.split(".")[0]
            if mod in suspicious_imports:
                self._record(node, f"suspicious import '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        suspicious_imports = {"ctypes", "cffi", "subprocess"}
        mod = (node.module or "").split(".")[0]
        if mod in suspicious_imports:
            self._record(node, f"suspicious from-import '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        # exec(...), eval(...)
        if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
            self._record(node, f"suspicious call '{node.func.id}'")

        # os.system(...)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "os" and node.func.attr == "system":
                    self._record(node, "suspicious call 'os.system'")
        self.generic_visit(node)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run untrusted connector files inside Kata Containers securely."
    )
    parser.add_argument(
        "connector",
        help="Path to connector JSON file, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Optional input JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--tmpdir",
        default=None,
        help="Base directory for run dirs (default: system temp).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when static AST safety scan finds suspicious patterns.",
    )
    parser.add_argument(
        "--keep-run-dir",
        action="store_true",
        help="Do not remove run directory after execution.",
    )
    return parser.parse_args()



def read_text_from_source(path_or_dash: str, stdin_cache: str | None) -> tuple[str, str | None]:
    if path_or_dash == "-":
        if stdin_cache is not None:
            raise OrchestratorError(
                "stdin already consumed; cannot use '-' for both connector and input"
            )
        data = sys.stdin.read()
        if not data.strip():
            raise OrchestratorError("stdin is empty")
        return data, data

    path = Path(path_or_dash)
    if not path.exists() or not path.is_file():
        raise OrchestratorError(f"file not found: {path_or_dash}")
    try:
        return path.read_text(encoding="utf-8"), stdin_cache
    except Exception as exc:
        raise OrchestratorError(f"failed reading file {path_or_dash}: {exc}") from exc



def parse_json_payload(payload_text: str, source_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"invalid JSON in {source_name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OrchestratorError(f"top-level JSON in {source_name} must be an object")
    return parsed



def resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise OrchestratorError(f"missing environment variable: {var_name}")
            return env_value

        return RE_PLACEHOLDER.sub(repl, value)

    if isinstance(value, list):
        return [resolve_env_placeholders(item) for item in value]

    if isinstance(value, dict):
        return {k: resolve_env_placeholders(v) for k, v in value.items()}

    return value



def merge_configuration(connector: dict[str, Any], input_obj: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(connector)
    base_config = merged.get("configuration")
    if base_config is None:
        base_config = {}
    if not isinstance(base_config, dict):
        raise OrchestratorError("connector.configuration must be an object when present")

    merged_config = dict(base_config)
    if input_obj is not None and "configuration" in input_obj:
        override = input_obj.get("configuration")
        if override is None:
            override = {}
        if not isinstance(override, dict):
            raise OrchestratorError("input.configuration must be an object when provided")
        merged_config.update(override)

    merged["configuration"] = merged_config
    return merged



def validate_connector_shape(connector: dict[str, Any]) -> None:
    if "files" not in connector:
        raise OrchestratorError("connector missing required top-level key: files")

    files = connector.get("files")
    if not isinstance(files, dict):
        raise OrchestratorError("connector.files must be an object map of path -> content")

    for path, content in files.items():
        if not isinstance(path, str) or not path:
            raise OrchestratorError("connector.files keys must be non-empty strings")
        if not isinstance(content, str):
            raise OrchestratorError(f"connector.files['{path}'] must be a string")

    requirements = connector.get("requirements")
    if requirements is not None:
        if not isinstance(requirements, list):
            raise OrchestratorError("connector.requirements must be an array of strings")
        for req in requirements:
            if not isinstance(req, str):
                raise OrchestratorError("all requirements entries must be strings")
            if not RE_REQUIREMENT.fullmatch(req):
                raise OrchestratorError(
                    f"invalid requirement '{req}': only package or package==version allowed"
                )

    runtime = connector.get("runtime")
    if runtime is not None and not isinstance(runtime, dict):
        raise OrchestratorError("connector.runtime must be an object when present")



def _normalize_type_descriptor(type_value: Any) -> dict[str, Any]:
    """
    Normalize a loose type descriptor from configurationTypes into JSON Schema fragment.

    Supported mappings:
      - string -> {"type":"string"}
      - number -> {"type":"number"}
      - boolean -> {"type":"boolean"}
      - array -> {"type":"array"}
      - object/dictionary -> {"type":"object"}
      - union via list or pipe-delimited string -> {"type":[...]} with mapped entries
    """
    map_atomic = {
        "string": "string",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
        "dictionary": "object",
        "null": "null",
        "integer": "integer",
    }

    def map_one(token: str) -> str:
        token_norm = token.strip().lower()
        if token_norm in map_atomic:
            return map_atomic[token_norm]
        return token_norm

    if isinstance(type_value, str):
        if "|" in type_value:
            types = [map_one(part) for part in type_value.split("|") if part.strip()]
            if not types:
                return {"type": "object"}
            return {"type": types}
        return {"type": map_one(type_value)}

    if isinstance(type_value, list):
        mapped = []
        for item in type_value:
            if isinstance(item, str):
                mapped.append(map_one(item))
        if mapped:
            return {"type": mapped}

    return {"type": "object"}



def build_configuration_schema(configuration_types: Any) -> dict[str, Any]:
    """Build JSON Schema for connector.configuration from connector.configurationTypes."""
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": True,
    }

    if configuration_types is None:
        return schema

    if not isinstance(configuration_types, dict):
        # Be permissive about dynamic data shape; skip schema details if malformed.
        return schema

    properties: dict[str, Any] = {}
    required: list[str] = []

    for field_name, descriptor in configuration_types.items():
        if not isinstance(field_name, str) or not field_name:
            continue

        if isinstance(descriptor, dict):
            type_fragment = _normalize_type_descriptor(descriptor.get("type", "object"))
            field_schema: dict[str, Any] = dict(type_fragment)

            # Optional pass-through of enum/default where present and JSON-compatible.
            if "enum" in descriptor and isinstance(descriptor["enum"], list):
                field_schema["enum"] = descriptor["enum"]
            if "default" in descriptor:
                field_schema["default"] = descriptor["default"]

            properties[field_name] = field_schema
            if bool(descriptor.get("required")):
                required.append(field_name)
        elif isinstance(descriptor, str) or isinstance(descriptor, list):
            properties[field_name] = _normalize_type_descriptor(descriptor)
        else:
            properties[field_name] = {"type": "object"}

    schema["properties"] = properties
    schema["required"] = required
    return schema



def validate_configuration(
    merged_configuration: dict[str, Any], configuration_types: Any
) -> None:
    if Draft202012Validator is None:
        raise OrchestratorError(
            "jsonschema is required but not installed. Install with: pip install jsonschema"
        )

    schema = build_configuration_schema(configuration_types)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(merged_configuration), key=lambda e: list(e.path))
    if not errors:
        return

    first = errors[0]
    loc = ".".join(str(x) for x in first.path) if first.path else "<root>"
    raise OrchestratorError(f"configuration validation error at {loc}: {first.message}")



def create_run_dirs(tmpdir: str | None) -> tuple[str, Path, Path]:
    run_id = uuid.uuid4().hex
    base = Path(tmpdir).resolve() if tmpdir else Path(tempfile.gettempdir())
    run_dir = Path(tempfile.mkdtemp(prefix=f"kata_run_{run_id}_", dir=str(base)))
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir, code_dir



def _safe_rel_path(path_str: str) -> Path:
    rel = Path(path_str)
    if rel.is_absolute():
        raise OrchestratorError(f"file path must be relative: {path_str}")
    normalized_parts = []
    for part in rel.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise OrchestratorError(f"path traversal not allowed in files key: {path_str}")
        normalized_parts.append(part)
    if not normalized_parts:
        raise OrchestratorError(f"invalid empty file path in files key: {path_str}")
    return Path(*normalized_parts)



def write_connector_files(code_dir: Path, files_map: dict[str, str]) -> None:
    for rel_path_str, content in files_map.items():
        rel = _safe_rel_path(rel_path_str)
        dest = code_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")



def write_requirements_file(code_dir: Path, requirements: list[str] | None) -> Path | None:
    if not requirements:
        return None

    validated: list[str] = []
    for req in requirements:
        if not RE_REQUIREMENT.fullmatch(req):
            raise OrchestratorError(
                f"invalid requirement '{req}': only package or package==version allowed"
            )
        validated.append(req)

    req_path = code_dir / "requirements.txt"
    req_path.write_text("\n".join(validated) + "\n", encoding="utf-8")
    return req_path



def detect_entrypoint(connector: dict[str, Any], code_dir: Path) -> str:
    runtime = connector.get("runtime") or {}
    entrypoint = runtime.get("entryPoint", DEFAULT_ENTRYPOINT)
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise OrchestratorError("runtime.entryPoint must be a non-empty string when present")

    rel = _safe_rel_path(entrypoint)
    full_path = code_dir / rel
    if not full_path.exists() or not full_path.is_file():
        raise OrchestratorError(f"entrypoint not found: {entrypoint}")
    return rel.as_posix()



def generate_seccomp_profile(run_dir: Path) -> Path:
    seccomp = {
        "defaultAction": "SCMP_ACT_ERRNO",
        "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86", "SCMP_ARCH_X32"],
        "archMap": [
            {
                "architecture": "SCMP_ARCH_X86_64",
                "subArchitectures": ["SCMP_ARCH_X86", "SCMP_ARCH_X32"],
            }
        ],
        "syscalls": [
            {
                "names": ["execve", "execveat"],
                "action": "SCMP_ACT_ALLOW",
                "args": [],
                "comment": "Allow process execution only",
                "includes": {},
                "excludes": {},
            },
            {
                "names": [
                    "fork",
                    "vfork",
                    "clone",
                    "clone3",
                    "ptrace",
                    "process_vm_readv",
                    "process_vm_writev",
                    "mount",
                    "setns",
                    "bpf",
                    "init_module",
                    "delete_module",
                    "reboot",
                    "open_by_handle_at",
                ],
                "action": "SCMP_ACT_ERRNO",
                "args": [],
                "comment": "Explicitly blocked high-risk syscalls",
                "includes": {},
                "excludes": {},
            },
        ],
    }
    seccomp_path = run_dir / "seccomp.json"
    seccomp_path.write_text(json.dumps(seccomp, indent=2), encoding="utf-8")
    return seccomp_path



def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def _flatten_env(prefix: str, obj: Any) -> dict[str, str]:
    """Flatten only scalar values for convenience env vars; avoids arbitrary deep expansion."""
    out: dict[str, str] = {}
    if not isinstance(obj, dict):
        return out

    for key, value in obj.items():
        if not isinstance(key, str):
            continue
        env_key = f"{prefix}_{re.sub(r'[^A-Za-z0-9_]', '_', key).upper()}"
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[env_key] = "" if value is None else str(value)
    return out



def build_bootstrap_script(run_dir: Path, entrypoint_rel: str) -> Path:
    script = f'''#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

code_dir = Path('/run/code')
requirements_path = code_dir / 'requirements.txt'
entrypoint = code_dir / {entrypoint_rel!r}

if requirements_path.exists():
    pip_cmd = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--user',
        '--no-cache-dir',
        '--disable-pip-version-check',
        '-r',
        str(requirements_path),
    ]
    pip_proc = subprocess.run(pip_cmd)
    if pip_proc.returncode != 0:
        sys.exit(pip_proc.returncode)

run_cmd = [sys.executable, str(entrypoint)]
proc = subprocess.run(run_cmd)
sys.exit(proc.returncode)
'''
    bootstrap_path = run_dir / "bootstrap.py"
    bootstrap_path.write_text(script, encoding="utf-8")
    return bootstrap_path



def truncate_text(data: bytes) -> str:
    if len(data) <= MAX_LOG_BYTES:
        return data.decode("utf-8", errors="replace")
    head = data[:MAX_LOG_BYTES]
    return head.decode("utf-8", errors="replace") + "\n...[truncated]"



def scan_python_files(code_dir: Path, strict: bool) -> tuple[list[str], list[str]]:
    result = SecurityScanResult()
    parse_errors: list[str] = []

    for py_file in code_dir.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except Exception as exc:
            parse_errors.append(f"{py_file}: failed to parse AST: {exc}")
            continue
        visitor = SuspiciousPythonVisitor(py_file.relative_to(code_dir), result)
        visitor.visit(tree)

    findings = result.findings + parse_errors
    if strict and findings:
        raise OrchestratorError("strict AST scan failed: " + "; ".join(findings))
    return result.findings, parse_errors



def build_docker_command(
    run_id: str,
    run_dir: Path,
    seccomp_path: Path,
    timeout_ms: int,
    memory_mb: int,
    cpus: float,
    pids_limit: int,
    has_input: bool,
    env_flattened: dict[str, str],
    image: str = DEFAULT_IMAGE,
) -> list[str]:
    container_name = f"kata-{run_id[:12]}"

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--runtime=kata",
        "--read-only",
        "--tmpfs",
        "/workspace:size=1G,mode=1777",
        "--mount",
        f"type=bind,src={run_dir},dst=/run,readonly",
        "--user",
        "65532:65532",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(pids_limit),
        "--memory",
        f"{memory_mb}m",
        "--cpus",
        str(cpus),
        "--security-opt",
        f"seccomp={seccomp_path}",
        "-e",
        "HOME=/workspace",
        "-e",
        "PYTHONUSERBASE=/workspace/.local",
        "-e",
        "PATH=/workspace/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "CONNECTOR_JSON_PATH=/run/connector.json",
    ]

    if has_input:
        cmd.extend(["-e", "INPUT_JSON_PATH=/run/input.json"])

    for env_k, env_v in env_flattened.items():
        cmd.extend(["-e", f"{env_k}={env_v}"])

    cmd.extend([
        image,
        "python",
        "/run/bootstrap.py",
    ])

    # timeout_ms argument retained for call-site parity with runtime contract.
    _ = timeout_ms
    return cmd



def run_container_with_timeout(
    cmd: list[str],
    run_id: str,
    timeout_ms: int,
) -> tuple[int, str, str]:
    container_name = f"kata-{run_id[:12]}"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout_ms / 1000)
    except subprocess.TimeoutExpired as exc:
        # Best effort kill by container name, then local process.
        try:
            subprocess.run(
                ["docker", "kill", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        finally:
            proc.kill()
            stdout_b, stderr_b = proc.communicate()
        raise TimeoutError(
            f"sandbox exceeded timeout of {timeout_ms} ms"
        ) from exc

    exit_code = proc.returncode if proc.returncode is not None else 1
    return exit_code, truncate_text(stdout_b), truncate_text(stderr_b)



def parse_result_from_stdout(stdout: str) -> Any:
    stripped = stdout.strip()
    if not stripped:
        return ""

    try:
        parsed = json.loads(stripped)
    except Exception:
        return stdout

    if isinstance(parsed, dict):
        return parsed
    return stdout



def json_print(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")



def main() -> int:
    args = parse_args()
    start = time.monotonic()
    run_id = uuid.uuid4().hex
    run_dir: Path | None = None

    memory_mb = DEFAULT_MEMORY_MB
    cpus = DEFAULT_CPUS
    pids_limit = DEFAULT_PIDS_LIMIT
    timeout_ms = DEFAULT_TIMEOUT_MS

    stdout_log = ""
    stderr_log = ""
    exit_code: int | None = None
    warnings: list[str] = []

    try:
        if args.connector == "-" and args.input_path == "-":
            raise OrchestratorError("connector and --input cannot both use stdin ('-')")

        stdin_cache: str | None = None

        connector_text, stdin_cache = read_text_from_source(args.connector, stdin_cache)
        connector = parse_json_payload(connector_text, "connector")
        connector = resolve_env_placeholders(connector)

        input_obj: dict[str, Any] | None = None
        if args.input_path is not None:
            input_text, stdin_cache = read_text_from_source(args.input_path, stdin_cache)
            input_obj = parse_json_payload(input_text, "input")

        merged_connector = merge_configuration(connector, input_obj)

        validate_connector_shape(merged_connector)
        validate_configuration(
            merged_connector.get("configuration", {}),
            merged_connector.get("configurationTypes"),
        )

        run_id, run_dir, code_dir = create_run_dirs(args.tmpdir)

        files_map = merged_connector["files"]
        write_connector_files(code_dir, files_map)
        write_requirements_file(code_dir, merged_connector.get("requirements"))
        entrypoint_rel = detect_entrypoint(merged_connector, code_dir)

        findings, parse_errors = scan_python_files(code_dir, strict=args.strict)
        warnings.extend(findings)
        warnings.extend(parse_errors)

        connector_path = run_dir / "connector.json"
        write_json_file(connector_path, merged_connector)

        if input_obj is not None:
            write_json_file(run_dir / "input.json", input_obj)

        seccomp_path = generate_seccomp_profile(run_dir)
        build_bootstrap_script(run_dir, entrypoint_rel)

        runtime = merged_connector.get("runtime") if isinstance(merged_connector.get("runtime"), dict) else {}
        if runtime:
            timeout_ms = int(runtime.get("timeoutMs", DEFAULT_TIMEOUT_MS))
            memory_mb = int(runtime.get("memoryMb", DEFAULT_MEMORY_MB))
            cpus = float(runtime.get("cpus", DEFAULT_CPUS))

        if timeout_ms <= 0:
            timeout_ms = DEFAULT_TIMEOUT_MS
        if memory_mb <= 0:
            memory_mb = DEFAULT_MEMORY_MB
        if cpus <= 0:
            cpus = DEFAULT_CPUS

        env_flattened = {}
        env_flattened.update(_flatten_env("AUTH", merged_connector.get("auth")))
        env_flattened.update(_flatten_env("CONFIG", merged_connector.get("configuration")))

        docker_cmd = build_docker_command(
            run_id=run_id,
            run_dir=run_dir,
            seccomp_path=seccomp_path,
            timeout_ms=timeout_ms,
            memory_mb=memory_mb,
            cpus=cpus,
            pids_limit=pids_limit,
            has_input=input_obj is not None,
            env_flattened=env_flattened,
        )

        exit_code, stdout_log, stderr_log = run_container_with_timeout(
            docker_cmd,
            run_id=run_id,
            timeout_ms=timeout_ms,
        )

        duration_ms = int((time.monotonic() - start) * 1000)

        if exit_code == 0:
            result = parse_result_from_stdout(stdout_log)
            out = {
                "ok": True,
                "result": result,
                "meta": {
                    "run_id": run_id,
                    "exit_code": 0,
                    "duration_ms": duration_ms,
                    "memory_limit_mb": memory_mb,
                    "pids_limit": pids_limit,
                },
            }
            if warnings:
                out["meta"]["warnings"] = warnings
            json_print(out)
            return 0

        out = {
            "ok": False,
            "error": f"sandbox exited with code {exit_code}",
            "meta": {
                "run_id": run_id,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "memory_limit_mb": memory_mb,
                "pids_limit": pids_limit,
            },
            "logs": {
                "stdout": stdout_log,
                "stderr": stderr_log,
            },
        }
        if warnings:
            out["meta"]["warnings"] = warnings
        json_print(out)
        return 1

    except TimeoutError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        out = {
            "ok": False,
            "error": str(exc),
            "meta": {
                "run_id": run_id,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "memory_limit_mb": memory_mb,
                "pids_limit": pids_limit,
            },
            "logs": {
                "stdout": stdout_log,
                "stderr": stderr_log,
            },
        }
        if warnings:
            out["meta"]["warnings"] = warnings
        json_print(out)
        return 1

    except OrchestratorError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        out = {
            "ok": False,
            "error": str(exc),
            "meta": {
                "run_id": run_id,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "memory_limit_mb": memory_mb,
                "pids_limit": pids_limit,
            },
            "logs": {
                "stdout": stdout_log,
                "stderr": stderr_log,
            },
        }
        if warnings:
            out["meta"]["warnings"] = warnings
        json_print(out)
        return 1

    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        out = {
            "ok": False,
            "error": f"unexpected error: {exc}",
            "meta": {
                "run_id": run_id,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "memory_limit_mb": memory_mb,
                "pids_limit": pids_limit,
            },
            "logs": {
                "stdout": stdout_log,
                "stderr": stderr_log,
            },
        }
        if warnings:
            out["meta"]["warnings"] = warnings
        json_print(out)
        return 1

    finally:
        if run_dir is not None and not args.keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
