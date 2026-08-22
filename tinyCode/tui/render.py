"""Input-line styling for the prompt_toolkit half of the terminal UI."""

from prompt_toolkit.styles import Style


STYLE = Style.from_dict(
    {
        "prompt": "bold green",
        "warning": "bold fg:#ffaa00",
    }
)
