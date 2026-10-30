"""Operator-owned settings for local workbenches and published demonstrations."""
from dataclasses import dataclass
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import yaml


DEMO_MESSAGE = "Official demo only. Training, testing and prediction require your own deployment."


def _origin(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("public_origin must be an HTTP(S) origin or null")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment or any(c.isspace() for c in value)
            or "\\" in value):
        raise ValueError("public_origin must contain only scheme, host and optional port")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("public_origin has an invalid port") from exc
    host = parsed.hostname.lower()
    try:
        if ipaddress.ip_address(host).version != 4:
            raise ValueError("public_origin currently supports IPv4 and DNS names only")
    except ValueError:
        if ":" in host:
            raise ValueError("public_origin currently supports IPv4 and DNS names only") from None
    authority = host + (f":{port}" if port is not None and port != {"http": 80, "https": 443}[parsed.scheme] else "")
    return f"{parsed.scheme}://{authority}"


@dataclass(frozen=True)
class WebGuiSettings:
    mode: str = "local"
    public_origin: str | None = None

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in {"local", "demo"}:
            raise ValueError("WebGUI mode must be local or demo")
        object.__setattr__(self, "public_origin", _origin(self.public_origin))

    @property
    def allow_task_execution(self):
        return self.mode == "local"

    def capabilities(self):
        return {"mode": self.mode, "allow_task_execution": self.allow_task_execution,
                "allow_server_mutations": self.allow_task_execution,
                "message": "" if self.allow_task_execution else DEMO_MESSAGE}


def load_settings(repository_root, config_path=None):
    root = Path(repository_root).resolve()
    path = Path(config_path).expanduser() if config_path is not None else root / "webgui.local.yaml"
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        if config_path is not None:
            raise ValueError(f"WebGUI configuration not found: {path}")
        return WebGuiSettings()
    try:
        if path.stat().st_size > 16 * 1024:
            raise ValueError("WebGUI configuration is too large")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to read WebGUI configuration: {path}") from exc
    if not isinstance(data, dict) or set(data) - {"mode", "public_origin"}:
        raise ValueError("WebGUI configuration accepts only mode and public_origin")
    return WebGuiSettings(**data)
