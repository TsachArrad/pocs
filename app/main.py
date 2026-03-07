import os
import copy
import json
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AzureOpenAI
import requests as http_requests

# Import validation functions from engine.py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import validate_connector_shape, validate_configuration, OrchestratorError

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



def _load_prompt_template() -> str:
    """Load the connector JSON schema prompt from fixtures/how to prompt.txt"""
    prompt_path = Path(__file__).parent.parent / "fixtures" / "how to prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Prompt template file not found")


def _validate_connector_json(connector_json: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate connector JSON using engine.py validators.
    Returns (is_valid, error_message)
    """
    try:
        # Validate shape (files, requirements, runtime)
        validate_connector_shape(connector_json)
        
        # Validate configuration if present
        configuration = connector_json.get("configuration", {})
        configuration_types = connector_json.get("configurationTypes")
        if configuration and configuration_types:
            validate_configuration(configuration, configuration_types)
        
        return True, None
    except OrchestratorError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract JSON object from text, handling markdown code blocks."""
    # Try to find JSON within markdown code blocks
    json_patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
        r"\{[^}]*\}",
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _get_client() -> AzureOpenAI:
    api_key = 'DAMxRHeLOWlar0mJcxBS0McuJw54EMcHYwm0MNVboV8l6vFIkB3CJQQJ99BKACF24PCXJ3w3AAAAACOGO7qx'
    endpoint = 'https://genor-prod-uaenorth-oai.cognitiveservices.azure.com'
    api_version = '2024-04-01-preview'

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


@app.get("/get-file/{file_name}")
def get_file(file_name: str) -> dict[str, Any]:
    url = f"https://localhost:62925/api/Connector/{file_name}"
    try:
        resp = http_requests.get(url, verify=False, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.ConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Connector API: {exc}") from exc
    except http_requests.exceptions.HTTPError as exc:
        raise HTTPException(status_code=resp.status_code, detail=resp.text) from exc
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Response is not valid JSON")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch file: {exc}") from exc

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    model = payload.model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    print(f"Using model: {model}")
    if not model:
        raise HTTPException(status_code=500, detail="AZURE_OPENAI_DEPLOYMENT is not set")

    default_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    default_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

    temperature = payload.temperature if payload.temperature is not None else default_temperature
    max_tokens = payload.max_tokens if payload.max_tokens is not None else default_max_tokens

    try:
        # Load the connector JSON schema prompt template
        prompt_template = _load_prompt_template()
        
        
        
        
        
        # Prepare messages: system prompt + user messages
        enhanced_messages = [
            {"role": "system", "content": prompt_template}
        ]
        enhanced_messages.extend([m.model_dump() for m in payload.messages])
        print(prompt_template)
        client = _get_client()
        
        # First call: Generate the connector
        print("Generating connector...")
        completion = client.chat.completions.create(
            model=model or 'gpt-5',
            messages=enhanced_messages,
            response_format={"type": "json_object"},
            max_completion_tokens=10000,
        )
        content = completion.choices[0].message.content
        
        print(f"Generated content length: {len(content) if content else 0}")
        if not content:
            raise HTTPException(status_code=502, detail="Model returned empty content")
        
        # Extract JSON from the response
        connector_json = _extract_json_from_text(content)
        
        if not connector_json:
            raise HTTPException(status_code=502, detail="Could not extract valid JSON from response")
        
        return ChatResponse(model=model, reply=content)
        # Validate using engine.py
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {exc}") from exc


class AddToStorageRequest(BaseModel):
    Content: str = Field(min_length=1)
    Name: str = Field(min_length=1)


@app.post("/add-to-storage")
def add_to_storage(payload: AddToStorageRequest) -> dict[str, Any]:
    url = "https://localhost:62925/api/Connector/add-to-storage"
    try:
        resp = http_requests.post(
            url,
            json=payload.model_dump(),
            verify=False,
            timeout=30,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"status": resp.status_code, "body": resp.text}
    except http_requests.exceptions.ConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach storage API: {exc}") from exc
    except http_requests.exceptions.HTTPError as exc:
        raise HTTPException(status_code=resp.status_code, detail=resp.text) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage request failed: {exc}") from exc


@app.post("/execute-connector", response_model=ExecuteConnectorResponse)
def execute_connector(payload: ExecuteConnectorRequest) -> ExecuteConnectorResponse:
    try:
        return _execute_connector_in_docker(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Connector execution engine failed: {exc}") from exc


@app.get("/validate-connector/{file_name:path}")
def validate_connector(file_name: str) -> dict[str, Any]:
    """
    1. Fetch connector JSON from https://localhost:62925/api/Connector/get-file/{file_name}
    2. Validate it using engine.py validators
    3. Return validation result
    """
    # Step 1: Fetch the connector JSON
    print(f"Fetching connector JSON for validation: {file_name}")
    url = f"https://localhost:62925/api/Connector/get-file/{file_name}"
    try:
        resp = http_requests.get(url, verify=False, timeout=600)
        resp.raise_for_status()
        connector_json = resp.json()
    except http_requests.exceptions.ConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Connector API: {exc}") from exc
    except http_requests.exceptions.HTTPError as exc:
        raise HTTPException(status_code=resp.status_code, detail=resp.text) from exc
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Response is not valid JSON")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch connector: {exc}") from exc
    
    # Step 2: Validate using engine.py validators
    print(f"Validating connector: {file_name}")
    is_valid, error_message = _validate_connector_json(connector_json)
    
    # Step 3: Return validation result
    if is_valid:
        return {
            "ok": True,
            "valid": True,
            "file_name": file_name,
            "message": "Connector JSON is valid"
        }
    else:
        return {
            "ok": False,
            "valid": False,
            "file_name": file_name,
            "error": error_message
        }


@app.get("/run-engine/{file_name:path}")
def run_engine(file_name: str) -> dict[str, Any]:
    """
    1. Fetch connector JSON from https://localhost:62925/api/Connector/{file_name}
    2. Write it to a temp file
    3. Run: python engine.py <temp_file>
    4. Return engine.py's JSON output
    """
    # Step 1: Fetch the connector JSON
    print(f"Fetching connector JSON for file: {file_name}")
    url = f"https://localhost:62925/api/Connector/get-file/{file_name}"
    try:
        resp = http_requests.get(url, verify=False, timeout=600)
        resp.raise_for_status()
        connector_json = resp.json()
    except http_requests.exceptions.ConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Connector API: {exc}") from exc
    except http_requests.exceptions.HTTPError as exc:
        raise HTTPException(status_code=resp.status_code, detail=resp.text) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch connector: {exc}") from exc

    # Step 2: Write to temp file and run engine.py
    engine_path = Path(__file__).parent.parent / "engine.py"
    if not engine_path.exists():
        raise HTTPException(status_code=500, detail="engine.py not found")

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(connector_json, tmp)
            tmp_path = tmp.name
        print(f"Fetched JSON for file: {file_name}")
        # Step 3: Run engine.py as subprocess
        result = subprocess.run(
            [sys.executable, str(engine_path), tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Step 4: Parse and return engine.py output
    stdout_text = result.stdout.strip() if result.stdout else ""
    stderr_text = result.stderr.strip() if result.stderr else ""

    parsed = _parse_json_if_possible(stdout_text)
    if isinstance(parsed, dict):
        return parsed

    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": stdout_text or None,
        "stderr": stderr_text or None,
    }


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
