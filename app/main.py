import os
import copy
import json
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import PurePosixPath
from typing import Any
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AzureOpenAI

load_dotenv()

app = FastAPI(title="OpenAI Python Server", version="1.0.0")

IGNORED_TOP_LEVEL_KEYS = {
    "schemaVersion",
    "connectorType",
    "displayName",
    "description",
    "version",
    "notes",
    "files",
}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)


class ChatResponse(BaseModel):
    model: str
    reply: str


class ExecuteConnectorRequest(BaseModel):
    connector: dict[str, Any]
    configuration_overrides: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_ms: int | None = Field(default=None, ge=100, le=300000)
    network_mode: Literal["bridge", "none"] | None = None


class ExecuteConnectorResponse(BaseModel):
    ok: bool
    result: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    error: str | None = None
    exit_code: int
    duration_ms: int
    stdout: str | None = None
    stderr: str | None = None



def _get_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key:
        raise HTTPException(status_code=500, detail="AZURE_OPENAI_API_KEY is not set")
    if not endpoint:
        raise HTTPException(status_code=500, detail="AZURE_OPENAI_ENDPOINT is not set")

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def _validate_relative_file_path(path_str: str) -> None:
    if not path_str:
        raise HTTPException(status_code=400, detail="files keys cannot be empty")

    file_path = PurePosixPath(path_str)
    if file_path.is_absolute() or ".." in file_path.parts:
        raise HTTPException(status_code=400, detail=f"Invalid file path in connector.files: {path_str}")


def _validate_connector(connector: dict[str, Any]) -> None:
    runtime = connector.get("runtime")
    if not isinstance(runtime, dict):
        raise HTTPException(status_code=400, detail="connector.runtime must be an object")

    language = runtime.get("language")
    if language != "python":
        raise HTTPException(status_code=400, detail="Only runtime.language='python' is currently supported")

    entry_point = runtime.get("entryPoint")
    if not isinstance(entry_point, str) or not entry_point.strip():
        raise HTTPException(status_code=400, detail="connector.runtime.entryPoint is required")

    dependencies = runtime.get("dependencies")
    if dependencies is not None:
        _normalize_runtime_dependencies(dependencies)

    files = connector.get("files")
    if not isinstance(files, dict) or not files:
        raise HTTPException(status_code=400, detail="connector.files must be a non-empty object")

    _validate_relative_file_path(entry_point)

    for name, content in files.items():
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="connector.files keys must be strings")
        _validate_relative_file_path(name)
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail=f"connector.files['{name}'] must be a string")

    if entry_point not in files:
        raise HTTPException(status_code=400, detail=f"Entry point '{entry_point}' not found in connector.files")


def _parse_json_if_possible(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _normalize_runtime_dependencies(raw_dependencies: Any) -> list[str]:
    def _validate_name(name: Any) -> str:
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=400, detail="runtime.dependencies name must be a non-empty string")
        name = name.strip()
        if not re.match(r"^[A-Za-z0-9_.-]+$", name):
            raise HTTPException(status_code=400, detail=f"Invalid dependency name: {name}")
        return name

    def _validate_version(version: Any) -> str:
        if version is None:
            return ""
        if not isinstance(version, str) or not version.strip():
            raise HTTPException(status_code=400, detail="runtime.dependencies version must be a string when provided")
        version = version.strip()
        if " " in version:
            raise HTTPException(status_code=400, detail=f"Invalid dependency version: {version}")
        return version

    normalized: list[str] = []

    if isinstance(raw_dependencies, dict):
        for name, version in raw_dependencies.items():
            dep_name = _validate_name(name)
            dep_version = _validate_version(version)
            normalized.append(f"{dep_name}=={dep_version}" if dep_version else dep_name)
        return sorted(normalized)

    if isinstance(raw_dependencies, list):
        for item in raw_dependencies:
            if isinstance(item, str):
                dep_name = _validate_name(item)
                normalized.append(dep_name)
                continue

            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=400,
                    detail="runtime.dependencies list items must be strings or objects with name/version",
                )

            dep_name = _validate_name(item.get("name"))
            dep_version = _validate_version(item.get("version"))
            normalized.append(f"{dep_name}=={dep_version}" if dep_version else dep_name)

        return sorted(set(normalized))

    raise HTTPException(
        status_code=400,
        detail="runtime.dependencies must be an array or object",
    )


def _path_exists(obj: dict[str, Any], dotted_path: str) -> bool:
    current: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _extract_direct_connector_paths(code: str) -> set[str]:
    paths: set[str] = set()

    direct_patterns = [
        r"\bconnector\.([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)",
        r"\bcontext\.connector\.([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)",
        r"\bconnector\[['\"]([A-Za-z_$][\w$]*)['\"]\]",
        r"\bcontext\.connector\[['\"]([A-Za-z_$][\w$]*)['\"]\]",
        r"\bcontext\[['\"]connector['\"]\]\[['\"]([A-Za-z_$][\w$]*)['\"]\]",
    ]

    for pattern in direct_patterns:
        for match in re.findall(pattern, code):
            if isinstance(match, tuple):
                candidate = next((group for group in match if group), "")
            else:
                candidate = match
            if candidate:
                paths.add(candidate)

    for match in re.finditer(r"\bconnector(?:\[['\"][A-Za-z_$][\w$]*['\"]\])+", code):
        keys = re.findall(r"\[['\"]([A-Za-z_$][\w$]*)['\"]\]", match.group(0))
        if keys:
            paths.add(".".join(keys))

    for match in re.finditer(r"\bcontext\[['\"]connector['\"]\](?:\[['\"][A-Za-z_$][\w$]*['\"]\])+", code):
        keys = re.findall(r"\[['\"]([A-Za-z_$][\w$]*)['\"]\]", match.group(0))
        if len(keys) >= 2 and keys[0] == "connector":
            paths.add(".".join(keys[1:]))

    return paths


def _strip_code_strings_and_comments(code: str) -> str:
    pattern = re.compile(
        r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|//[^\n]*|/\*[\s\S]*?\*/|#[^\n]*"
    )
    return pattern.sub(" ", code)


def _extract_alias_connector_paths(code: str) -> set[str]:
    paths: set[str] = set()

    alias_pattern = re.compile(
        r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:context\.)?connector\.([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)"
    )
    python_alias_pattern = re.compile(
        r"\b([A-Za-z_][\w]*)\s*=\s*(?:context\.)?connector\.([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)"
    )
    aliases: dict[str, str] = {}
    for alias_name, base_path in alias_pattern.findall(code):
        aliases[alias_name] = base_path
    for alias_name, base_path in python_alias_pattern.findall(code):
        aliases[alias_name] = base_path

    for alias_name, base_path in aliases.items():
        attr_pattern = re.compile(rf"\b{re.escape(alias_name)}\.([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)")
        bracket_pattern = re.compile(rf"\b{re.escape(alias_name)}\[['\"]([A-Za-z_$][\w$]*)['\"]\]")

        alias_refs = {ref for ref in attr_pattern.findall(code)}
        alias_refs.update(bracket_pattern.findall(code))

        if not alias_refs:
            paths.add(base_path)
            continue

        for ref in alias_refs:
            paths.add(f"{base_path}.{ref}")

    return paths


def _discover_referenced_connector_paths(files: dict[str, Any]) -> list[str]:
    all_paths: set[str] = set()

    for file_name, content in files.items():
        if not isinstance(content, str):
            continue
        code = _strip_code_strings_and_comments(content)
        all_paths.update(_extract_direct_connector_paths(code))
        all_paths.update(_extract_alias_connector_paths(code))

    normalized_paths: set[str] = set()
    for dotted_path in all_paths:
        root = dotted_path.split(".", 1)[0]
        if root in IGNORED_TOP_LEVEL_KEYS:
            continue
        normalized_paths.add(dotted_path)

    return sorted(normalized_paths)


def _required_top_level_keys_from_paths(paths: list[str]) -> list[str]:
    top_level = {path.split(".", 1)[0] for path in paths}
    top_level = {key for key in top_level if key not in IGNORED_TOP_LEVEL_KEYS}
    return sorted(top_level)


def _extract_required_env_keys(connector: dict[str, Any]) -> list[str]:
    required: set[str] = set()

    connector_text = json.dumps(connector)
    required.update(re.findall(r"\{\{env:([A-Z][A-Z0-9_]*)\}\}", connector_text))

    files = connector.get("files")
    if isinstance(files, dict):
        for content in files.values():
            if not isinstance(content, str):
                continue
            required.update(re.findall(r"\bos\.getenv\(\s*['\"]([A-Z][A-Z0-9_]*)['\"]", content))
            required.update(re.findall(r"\bos\.environ\[['\"]([A-Z][A-Z0-9_]*)['\"]\]", content))
            required.update(re.findall(r"\bos\.environ\.get\(\s*['\"]([A-Z][A-Z0-9_]*)['\"]", content))
            required.update(re.findall(r"\benviron\[['\"]([A-Z][A-Z0-9_]*)['\"]\]", content))
            required.update(re.findall(r"\benviron\.get\(\s*['\"]([A-Z][A-Z0-9_]*)['\"]", content))
            required.update(re.findall(r"\bprocess\.env\.([A-Z][A-Z0-9_]*)\b", content))
            required.update(re.findall(r"\bprocess\.env\[['\"]([A-Z][A-Z0-9_]*)['\"]\]", content))

    return sorted(required)


def _resolve_container_env(payload_env: dict[str, str], required_env_keys: list[str]) -> dict[str, str]:
    resolved = dict(payload_env)
    missing: list[str] = []

    for key in required_env_keys:
        if key in resolved:
            continue
        host_value = os.getenv(key)
        if host_value is None:
            missing.append(key)
        else:
            resolved[key] = host_value

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required environment variables for connector execution",
                "missing": missing,
            },
        )

    return resolved


def _execute_connector_in_docker(payload: ExecuteConnectorRequest) -> ExecuteConnectorResponse:
    connector = copy.deepcopy(payload.connector)
    _validate_connector(connector)

    configuration = connector.get("configuration")
    if configuration is not None and not isinstance(configuration, dict):
        raise HTTPException(status_code=400, detail="connector.configuration must be an object when provided")

    if payload.configuration_overrides:
        if configuration is None:
            connector["configuration"] = {}
            configuration = connector["configuration"]
        configuration.update(payload.configuration_overrides)

    referenced_paths = _discover_referenced_connector_paths(connector["files"])
    if not referenced_paths:
        raise HTTPException(
            status_code=400,
            detail="No connector JSON parameter references were found in files code",
        )

    required_top_level_keys = _required_top_level_keys_from_paths(referenced_paths)
    missing_top_level_keys = [key for key in required_top_level_keys if key not in connector]
    if missing_top_level_keys:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Connector is missing required top-level keys used by code",
                "missing": missing_top_level_keys,
            },
        )

    missing_paths = [path for path in referenced_paths if not _path_exists(connector, path)]
    if missing_paths:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing connector values referenced by code",
                "missing": missing_paths,
            },
        )

    runtime = connector["runtime"]
    entry_point = runtime["entryPoint"]
    python_dependencies = _normalize_runtime_dependencies(runtime.get("dependencies", []))
    timeout_ms = payload.timeout_ms or int(runtime.get("timeoutMs", 30000))
    memory_mb = int(runtime.get("memoryMb", 256))
    network_mode = payload.network_mode or "bridge"

    if memory_mb < 64 or memory_mb > 4096:
        raise HTTPException(status_code=400, detail="runtime.memoryMb must be between 64 and 4096")

    env_key_pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
    for env_key in payload.env:
        if not env_key_pattern.match(env_key):
            raise HTTPException(status_code=400, detail=f"Invalid env key: {env_key}")

    required_env_keys = _extract_required_env_keys(connector)
    container_env = _resolve_container_env(payload.env, required_env_keys)

    started_at = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="connector-run-") as temp_dir:
        run_dir = os.path.join(temp_dir, "run")
        code_dir = os.path.join(run_dir, "code")
        os.makedirs(code_dir, exist_ok=True)

        connector_json_path = os.path.join(run_dir, "connector.json")
        with open(connector_json_path, "w", encoding="utf-8") as connector_file:
            json.dump(connector, connector_file)

        for file_name, content in connector["files"].items():
            destination_path = os.path.join(code_dir, file_name)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            with open(destination_path, "w", encoding="utf-8") as output_file:
                output_file.write(content)

        mount_path = f"{run_dir}:/run:rw"
        container_entrypoint = f"/run/code/{entry_point}"

        command = [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "128",
            "--memory",
            f"{memory_mb}m",
            "--network",
            network_mode,
            "-v",
            mount_path,
            "-w",
            "/run/code",
        ]

        for key, value in container_env.items():
            command.extend(["-e", f"{key}={value}"])

        install_and_run_script_parts: list[str] = []
        if python_dependencies:
            quoted_dependencies = " ".join(shlex.quote(dep) for dep in python_dependencies)
            install_and_run_script_parts.append(
                f"python -m pip install --no-cache-dir --disable-pip-version-check {quoted_dependencies}"
            )

        install_and_run_script_parts.append(f"python {shlex.quote(container_entrypoint)}")
        install_and_run_script = " && ".join(install_and_run_script_parts)

        command.extend([
            "python:3.13-slim",
            "sh",
            "-lc",
            install_and_run_script,
        ])

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_ms / 1000,
                check=False,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="Docker CLI not found on server") from exc
        except subprocess.TimeoutExpired as exc:
            elapsed = int((time.perf_counter() - started_at) * 1000)
            return ExecuteConnectorResponse(
                ok=False,
                error=f"Connector execution timed out after {timeout_ms} ms",
                exit_code=124,
                duration_ms=elapsed,
                stdout=(exc.stdout or ""),
                stderr=(exc.stderr or ""),
            )

    elapsed = int((time.perf_counter() - started_at) * 1000)
    stdout_text = completed.stdout.strip() if completed.stdout else ""
    stderr_text = completed.stderr.strip() if completed.stderr else ""

    if completed.returncode == 0:
        payload_json = _parse_json_if_possible(stdout_text)
        if isinstance(payload_json, dict) and payload_json.get("ok") is True:
            return ExecuteConnectorResponse(
                ok=True,
                result=payload_json.get("result"),
                exit_code=0,
                duration_ms=elapsed,
                stdout=stdout_text,
                stderr=stderr_text or None,
            )

        return ExecuteConnectorResponse(
            ok=True,
            result=payload_json if payload_json is not None else stdout_text,
            exit_code=0,
            duration_ms=elapsed,
            stdout=stdout_text,
            stderr=stderr_text or None,
        )

    stderr_json = _parse_json_if_possible(stderr_text)
    error_message = None
    if isinstance(stderr_json, dict):
        error_message = stderr_json.get("error") if isinstance(stderr_json.get("error"), str) else None

    if not error_message:
        error_message = stderr_text or "Connector execution failed"

    return ExecuteConnectorResponse(
        ok=False,
        error=error_message,
        exit_code=completed.returncode,
        duration_ms=elapsed,
        stdout=stdout_text or None,
        stderr=stderr_text or None,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health2")
def health() -> dict[str, str]:
    x = "dasdsa"
    y =x + "sdfsdf"
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    model = payload.model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not model:
        raise HTTPException(status_code=500, detail="AZURE_OPENAI_DEPLOYMENT is not set")

    default_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    default_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

    temperature = payload.temperature if payload.temperature is not None else default_temperature
    max_tokens = payload.max_tokens if payload.max_tokens is not None else default_max_tokens

    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in payload.messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = completion.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="Model returned empty content")

        return ChatResponse(model=model, reply=content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {exc}") from exc


@app.post("/execute-connector", response_model=ExecuteConnectorResponse)
def execute_connector(payload: ExecuteConnectorRequest) -> ExecuteConnectorResponse:
    try:
        return _execute_connector_in_docker(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Connector execution engine failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path

    port = int(os.getenv("PORT", "8000"))
    project_root = Path(__file__).resolve().parents[1]
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        app_dir=str(project_root),
    )
