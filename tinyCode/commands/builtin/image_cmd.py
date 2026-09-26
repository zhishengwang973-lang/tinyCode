"""Attach a local or remote image to a new model turn."""

from tinyCode.commands.types import CommandMeta, CommandType, ParamHint, UIControl
from tinyCode.multimodal import ImageInputError, build_image_user_content


def create(ui: UIControl) -> CommandMeta:
    async def handler(args: list[str]) -> str | None:
        if not args:
            return "用法: /image [--detail low|high|original|auto] <路径或URL> [问题]"
        if not ui.supports_image_input():
            return (
                "当前模型不支持图片输入；请切换到视觉模型，"
                "如 deepseek-flash、gpt-4o 或 Claude 3+"
            )

        detail = "auto"
        remaining = list(args)
        if remaining[0] == "--detail":
            if len(remaining) < 3:
                return "用法: /image --detail <low|high|original|auto> <路径或URL> [问题]"
            detail = remaining[1]
            remaining = remaining[2:]
        elif remaining[0].startswith("--detail="):
            detail = remaining[0].split("=", 1)[1]
            remaining = remaining[1:]
        if not remaining:
            return "图片路径或 URL 不能为空"

        source = remaining[0]
        prompt = " ".join(remaining[1:])
        try:
            content, label = build_image_user_content(
                source,
                prompt,
                detail=detail,
                project_root=ui.get_image_attachment_root(),
            )
        except ImageInputError as exc:
            return str(exc)
        display_text = "\n".join(
            part for part in (f"[图片: {label}]", prompt.strip()) if part
        )
        if not prompt.strip():
            display_text += "\n请描述并分析这张图片。"
        if not ui.send_image_to_conversation(content, display_text):
            return "已有任务正在执行，暂时无法附加图片"
        return None

    return CommandMeta(
        name="image",
        aliases=["img"],
        description="发送本地图片或图片 URL 给视觉模型",
        usage="/image [--detail low|high|original|auto] <路径或URL> [问题]",
        cmd_type=CommandType.UI,
        params=[
            ParamHint("路径或URL", "JPEG / PNG / GIF / WebP", required=True),
            ParamHint("问题", "需要模型回答的问题"),
        ],
        handler=handler,
    )
