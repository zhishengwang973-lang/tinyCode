"""Select the configured terminal renderer without changing agent wiring."""

from __future__ import annotations

from typing import Any

from tinyCode.tui.app import TinyCodeTUI
from tinyCode.tui.fullscreen_textual import FullscreenTinyCodeTUI


def create_tui(*, ui_mode: str = "stream", **kwargs: Any) -> TinyCodeTUI:
    """Create the requested UI, falling back safely outside a real terminal."""
    if ui_mode == "fullscreen" and FullscreenTinyCodeTUI.supported():
        return FullscreenTinyCodeTUI(**kwargs)
    return TinyCodeTUI(**kwargs)


def fullscreen_supported() -> bool:
    return FullscreenTinyCodeTUI.supported()
