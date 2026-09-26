"""Safe, persistent image attachments for multimodal user messages."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from tinyCode.providers.base import Message, ProviderError


SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
SUPPORTED_DETAILS = {"low", "high", "original", "auto"}
MAX_INLINE_IMAGE_BYTES = 32 * 1024 * 1024
MAX_INLINE_TOTAL_BYTES = 32 * 1024 * 1024
MAX_IMAGE_URL_CHARS = 8192
IMAGE_TOKEN_ESTIMATE = 1024


class ImageInputError(ValueError):
    """A user-correctable image attachment error."""


_MACOS_CLIPBOARD_IMAGE_SCRIPT = r'''
on run argv
    set targetPath to item 1 of argv
    try
        set imageData to the clipboard as «class PNGf»
        set imageFormat to "png"
    on error
        try
            set imageData to the clipboard as TIFF picture
            set imageFormat to "tiff"
        on error
            return "no-image"
        end try
    end try

    set targetFile to open for access POSIX file targetPath with write permission
    try
        set eof targetFile to 0
        write imageData to targetFile
        close access targetFile
    on error errorMessage
        try
            close access targetFile
        end try
        error errorMessage
    end try
    return imageFormat
end run
'''


async def select_local_image() -> str | None:
    """Open the platform-native file chooser and return one selected path."""
    if sys.platform == "darwin":
        command = (
            "/usr/bin/osascript",
            "-e",
            'POSIX path of (choose file with prompt "选择要发送给 TinyCode 的图片")',
        )
    elif sys.platform.startswith("linux") and shutil.which("zenity"):
        command = (
            shutil.which("zenity") or "zenity",
            "--file-selection",
            "--title=选择要发送给 TinyCode 的图片",
            "--file-filter=图片 | *.jpg *.jpeg *.png *.gif *.webp",
        )
    elif sys.platform.startswith("linux") and shutil.which("kdialog"):
        command = (
            shutil.which("kdialog") or "kdialog",
            "--getopenfilename",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.gif *.webp)",
        )
    else:
        raise ImageInputError(
            "当前系统没有可用的图形文件选择器，请使用 /image <图片路径>"
        )

    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=300.0,
        )
    except asyncio.TimeoutError as exc:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        raise ImageInputError("图片文件选择超时") from exc
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        raise
    except OSError as exc:
        raise ImageInputError(f"无法打开图片文件选择器: {exc}") from exc

    if process.returncode != 0:
        # Native pickers use a non-zero status when the user presses Cancel.
        error = stderr.decode("utf-8", errors="replace").strip()
        if "canceled" in error.casefold() or "cancelled" in error.casefold():
            return None
        if sys.platform == "darwin" and process.returncode == 1:
            return None
        raise ImageInputError(error or "图片文件选择器异常退出")
    selected = stdout.decode("utf-8", errors="replace").strip()
    return selected or None


async def paste_clipboard_image(project_root: Path | None = None) -> str | None:
    """Persist an OS clipboard image and return its durable local path.

    A terminal paste only transports text. This function deliberately reads
    the native pasteboard when the TUI receives its explicit paste-image
    shortcut. ``None`` means the clipboard currently has no image flavor.
    """
    if sys.platform != "darwin":
        raise ImageInputError(
            "当前系统暂不支持直接读取图片剪贴板，请使用 + 选择图片"
        )

    root = (project_root or Path.cwd()).resolve()
    attachment_dir = root / ".tinyCode" / "attachments"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    raw_path = attachment_dir / f".clipboard-{uuid4().hex}.raw"
    converted_path = attachment_dir / f".clipboard-{uuid4().hex}.png"
    selected_path = raw_path
    try:
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                "/usr/bin/osascript",
                "-e",
                _MACOS_CLIPBOARD_IMAGE_SCRIPT,
                str(raw_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=10.0,
            )
        except asyncio.TimeoutError as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise ImageInputError("读取系统图片剪贴板超时") from exc
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except OSError as exc:
            raise ImageInputError(f"无法读取系统图片剪贴板: {exc}") from exc

        if process.returncode != 0:
            error = stderr.decode("utf-8", errors="replace").strip()
            raise ImageInputError(error or "读取系统图片剪贴板失败")
        image_format = stdout.decode("utf-8", errors="replace").strip().casefold()
        if image_format == "no-image":
            return None
        if image_format == "tiff":
            selected_path = converted_path
            await _convert_macos_tiff_to_png(raw_path, converted_path)
        elif image_format != "png":
            raise ImageInputError("系统剪贴板返回了无法识别的图片格式")

        content, _ = await asyncio.to_thread(
            build_image_user_content,
            str(selected_path),
            "",
            project_root=root,
        )
        image = content[-1].get("image_file", {})
        path = image.get("path") if isinstance(image, dict) else None
        if not isinstance(path, str) or not path:
            raise ImageInputError("保存剪贴板图片失败")
        return path
    finally:
        for temporary in (raw_path, converted_path):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


async def read_clipboard_text() -> str:
    """Read native clipboard text for paste-image shortcut fallback."""
    if sys.platform != "darwin":
        return ""
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/pbpaste",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3.0)
    except asyncio.TimeoutError:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        return ""
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        raise
    except OSError:
        return ""
    if process.returncode != 0:
        return ""
    return stdout.decode("utf-8", errors="replace")


async def _convert_macos_tiff_to_png(source: Path, destination: Path) -> None:
    process: asyncio.subprocess.Process | None = None
    stderr = b""
    try:
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/sips",
            "-s",
            "format",
            "png",
            str(source),
            "--out",
            str(destination),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
    except asyncio.TimeoutError as exc:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        raise ImageInputError("转换剪贴板图片超时") from exc
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        raise
    except OSError as exc:
        raise ImageInputError(f"转换剪贴板图片失败: {exc}") from exc
    if process is None or process.returncode != 0:
        error = stderr.decode("utf-8", errors="replace").strip()
        raise ImageInputError(error or "无法将剪贴板图片转换为 PNG")


def build_image_user_content(
    source: str,
    prompt: str,
    *,
    detail: str = "auto",
    project_root: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Build a lightweight user content list and a readable display label."""
    detail = detail.strip().lower()
    if detail not in SUPPORTED_DETAILS:
        raise ImageInputError("图片 detail 必须是 low、high、original 或 auto")
    source = source.strip()
    if not source:
        raise ImageInputError("图片路径或 URL 不能为空")
    text = prompt.strip() or "请描述并分析这张图片。"
    parsed = urlsplit(source)
    if parsed.scheme in {"http", "https"}:
        if len(source) > MAX_IMAGE_URL_CHARS:
            raise ImageInputError(
                f"图片 URL 不能超过 {MAX_IMAGE_URL_CHARS} 个字符"
            )
        image_block = {
            "type": "image_url",
            "image_url": {"url": source, "detail": detail},
        }
        label = parsed.path.rsplit("/", 1)[-1] or parsed.hostname or "远程图片"
    elif parsed.scheme:
        raise ImageInputError("图片地址只支持本地路径或 http(s) URL")
    else:
        original = Path(source).expanduser().resolve()
        if not original.is_file():
            raise ImageInputError(f"图片文件不存在: {original}")
        try:
            size = original.stat().st_size
        except OSError as exc:
            raise ImageInputError(f"无法读取图片信息: {exc}") from exc
        if size <= 0:
            raise ImageInputError("图片文件为空")
        if size > MAX_INLINE_IMAGE_BYTES:
            raise ImageInputError(
                "本地图片不能超过 32 MiB；更大的图片需要使用 DeepSeek Files API"
            )
        try:
            with original.open("rb") as handle:
                header = handle.read(16)
        except OSError as exc:
            raise ImageInputError(f"无法读取图片: {exc}") from exc
        media_type = detect_image_media_type(header)
        if media_type is None:
            raise ImageInputError("仅支持实际内容为 JPEG、PNG、GIF 或 WebP 的图片")
        root = (project_root or Path.cwd()).resolve()
        attachment_dir = root / ".tinyCode" / "attachments"
        attachment_dir.mkdir(parents=True, exist_ok=True)
        digest = _sha256_file(original)
        destination = attachment_dir / f"{digest}{SUPPORTED_IMAGE_TYPES[media_type]}"
        needs_copy = not destination.exists()
        if not needs_copy:
            try:
                needs_copy = _sha256_file(destination) != digest
            except ImageInputError:
                needs_copy = True
        if needs_copy:
            temporary = attachment_dir / (
                f".{digest}.{os.getpid()}.{uuid4().hex}.tmp"
            )
            try:
                with original.open("rb") as source_handle, temporary.open("xb") as out:
                    while chunk := source_handle.read(1024 * 1024):
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, destination)
            except OSError as exc:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ImageInputError(f"保存图片附件失败: {exc}") from exc
        image_block = {
            "type": "image_file",
            "image_file": {
                "path": str(destination),
                "media_type": media_type,
                "detail": detail,
                "size": size,
                "sha256": digest,
                "name": original.name,
            },
        }
        label = original.name
    return ([{"type": "text", "text": text}, image_block], label)


def detect_image_media_type(header: bytes) -> str | None:
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def extract_text_content(content: object) -> str:
    """Return user-visible text without serializing image payloads."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
        and block.get("type") in {"text", "input_text"}
        and isinstance(block.get("text"), str)
    ]
    return "\n".join(part for part in parts if part).strip()


def describe_user_content(content: object) -> str:
    text = extract_text_content(content)
    if not isinstance(content, list):
        return text
    labels: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image_file":
            image = block.get("image_file")
            path = image.get("path", "") if isinstance(image, dict) else ""
            name = image.get("name", "") if isinstance(image, dict) else ""
            labels.append(
                name
                if isinstance(name, str) and name
                else Path(path).name if isinstance(path, str) and path else "图片"
            )
        elif block.get("type") == "image_url":
            labels.append("远程图片")
    prefix = " ".join(f"[图片: {label}]" for label in labels)
    return "\n".join(part for part in (prefix, text) if part)


def materialize_deepseek_images(messages: list[Message]) -> list[Message]:
    """Convert persistent local-image references to DeepSeek image_url blocks."""
    result = deepcopy(messages)
    total_bytes = 0
    image_count = 0
    for message in result:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        has_image = any(
            isinstance(block, dict)
            and block.get("type") in {"image_file", "image_url"}
            for block in content
        )
        if has_image and message.get("role") != "user":
            raise ProviderError(
                "DeepSeek 图片只能出现在 user 消息中",
                code="invalid_image_role",
            )
        converted: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_file":
                converted.append(block)
                if isinstance(block, dict) and block.get("type") == "image_url":
                    image_count += 1
                continue
            image = block.get("image_file")
            if not isinstance(image, dict):
                raise ProviderError("图片附件结构无效", code="invalid_image")
            path = image.get("path")
            media_type = image.get("media_type")
            detail = image.get("detail", "auto")
            expected_digest = image.get("sha256")
            if (
                not isinstance(path, str)
                or media_type not in SUPPORTED_IMAGE_TYPES
                or detail not in SUPPORTED_DETAILS
                or not isinstance(expected_digest, str)
                or len(expected_digest) != 64
            ):
                raise ProviderError("图片附件元数据无效", code="invalid_image")
            file_path = Path(path)
            try:
                data = file_path.read_bytes()
            except OSError as exc:
                raise ProviderError(
                    f"无法读取图片附件 {file_path.name}: {exc}",
                    code="image_unavailable",
                ) from exc
            if detect_image_media_type(data[:16]) != media_type:
                raise ProviderError(
                    f"图片附件 {file_path.name} 的实际格式已变化",
                    code="invalid_image",
                )
            if hashlib.sha256(data).hexdigest() != expected_digest:
                raise ProviderError(
                    f"图片附件 {file_path.name} 内容已变化，请重新附加",
                    code="image_changed",
                )
            total_bytes += len(data)
            image_count += 1
            if total_bytes > MAX_INLINE_TOTAL_BYTES:
                raise ProviderError(
                    "当前请求的本地图片总大小超过 32 MiB；请新建会话或减少图片",
                    code="image_request_too_large",
                )
            encoded = base64.b64encode(data).decode("ascii")
            converted.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{encoded}",
                    "detail": detail,
                },
            })
        message["content"] = converted
    if image_count > 600:
        raise ProviderError(
            "DeepSeek 单个请求最多包含 600 张图片",
            code="too_many_images",
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ImageInputError(f"计算图片摘要失败: {exc}") from exc
    return digest.hexdigest()
