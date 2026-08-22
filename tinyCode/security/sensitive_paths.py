"""Conservative detection for local credential and secret paths."""

from __future__ import annotations

from pathlib import PurePath


_SENSITIVE_BASENAMES = frozenset({
    ".tinycode.yaml",
    ".tinycode.yml",
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
})

_SENSITIVE_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".kube"})


def is_sensitive_path(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    pure = PurePath(path)
    parts = tuple(part.casefold() for part in pure.parts)
    basename = pure.name.casefold()
    if basename in _SENSITIVE_BASENAMES or basename.startswith(".env."):
        return True
    if any(part in _SENSITIVE_DIRS for part in parts):
        return True
    # The global provider configuration is ~/.tinyCode/config.yaml.
    return basename == "config.yaml" and ".tinycode" in parts


def command_references_sensitive_path(command: object) -> bool:
    if not isinstance(command, str):
        return False
    folded = command.casefold()
    compact = folded.replace("\\", "/")
    if any(name in folded for name in _SENSITIVE_BASENAMES):
        return True
    if ".env." in folded or any(f"/{name}/" in f"/{compact}/" for name in _SENSITIVE_DIRS):
        return True
    if ".tinycode/config.yaml" in compact:
        return True
    # Environment dumping is a credential disclosure primitive, not a benign
    # inspection command.
    first = folded.strip().split(maxsplit=1)[0] if folded.strip() else ""
    return PurePath(first).name in {"env", "printenv", "set", "export"}
