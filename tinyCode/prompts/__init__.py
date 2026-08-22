"""Prompt management — modular system prompt, injections, environment context."""

from tinyCode.prompts.loader import PromptModule, load_modules, load_injection
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.prompts.environment import collect_environment

__all__ = [
    "PromptModule",
    "PromptBuilder",
    "PromptInjector",
    "collect_environment",
    "load_modules",
    "load_injection",
]
