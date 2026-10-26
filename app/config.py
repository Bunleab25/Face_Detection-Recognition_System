from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value.strip() != "" else default


def _resolve_project_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _load_dotenv_if_present() -> None:
    """Minimal .env loader (no external dependency).

    Loads key=value pairs from a .env file located at the repo root (one level above app/)
    or current working directory. Existing environment variables are not overwritten.
    """

    candidates = [
        Path.cwd() / ".env",
        PROJECT_ROOT / ".env",
    ]
    def should_override(key: str, new_value: str) -> bool:
        current = os.getenv(key)
        if current is None or current.strip() == "":
            return True

        # If a previously-set model path points to a missing file, prefer .env
        if key in {"YOLO_MODEL_PATH", "ARCFACE_CKPT_PATH"}:
            try:
                if not Path(current).exists() and Path(new_value).exists():
                    return True
            except OSError:
                return True

        return False

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and should_override(key, value):
                    os.environ[key] = value
        except OSError:
            pass
        break


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    yolo_model_path: str
    arcface_ckpt_path: str
    device: str
    yolo_conf: float
    input_size: int


def load_settings() -> Settings:
    _load_dotenv_if_present()
    return Settings(
        yolo_model_path=_resolve_project_path(
            _get_env_str("YOLO_MODEL_PATH", "models/yolo_face.pt")
        ),
        arcface_ckpt_path=_resolve_project_path(
            _get_env_str("ARCFACE_CKPT_PATH", "models/arcface.pth")
        ),
        device=_get_env_str("DEVICE", "cuda"),
        yolo_conf=_get_env_float("YOLO_CONF", 0.5),
        input_size=_get_env_int("ARCFACE_INPUT_SIZE", 112),
    )


settings = load_settings()
