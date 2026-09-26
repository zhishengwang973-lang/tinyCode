import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tinyCode.conversation.history import ConversationHistory
from tinyCode.multimodal import (
    IMAGE_TOKEN_ESTIMATE,
    ImageInputError,
    build_image_user_content,
    describe_user_content,
    materialize_anthropic_images,
    materialize_deepseek_images,
    materialize_openai_images,
    paste_clipboard_image,
    select_local_image,
)


_PNG = b"\x89PNG\r\n\x1a\n" + b"test-image-data"


class MultimodalInputTests(unittest.TestCase):
    def test_local_image_is_copied_and_persisted_as_lightweight_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "screen.png"
            source.write_bytes(_PNG)

            content, label = build_image_user_content(
                str(source), "分析这个界面", project_root=root,
            )

            self.assertEqual("screen.png", label)
            self.assertEqual("text", content[0]["type"])
            self.assertEqual("image_file", content[1]["type"])
            attachment = Path(content[1]["image_file"]["path"])
            self.assertTrue(attachment.is_file())
            self.assertEqual(_PNG, attachment.read_bytes())
            self.assertNotIn("base64", str(content))

    def test_materialization_uses_deepseek_image_url_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "screen.png"
            source.write_bytes(_PNG)
            content, _ = build_image_user_content(
                str(source), "识别文字", detail="low", project_root=root,
            )

            messages = materialize_deepseek_images([
                {"role": "user", "content": content},
            ])

            image = messages[0]["content"][1]
            self.assertEqual("image_url", image["type"])
            self.assertEqual("low", image["image_url"]["detail"])
            expected = base64.b64encode(_PNG).decode("ascii")
            self.assertEqual(
                f"data:image/png;base64,{expected}",
                image["image_url"]["url"],
            )
            self.assertEqual("image_file", content[1]["type"])

    def test_openai_and_anthropic_materialize_local_images_for_their_protocols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "screen.png"
            source.write_bytes(_PNG)
            content, _ = build_image_user_content(
                str(source), "识别文字", project_root=root,
            )
            messages = [{"role": "user", "content": content}]

            openai = materialize_openai_images(messages)
            anthropic = materialize_anthropic_images(messages)

            self.assertEqual("image_url", openai[0]["content"][1]["type"])
            image = anthropic[0]["content"][1]
            self.assertEqual("image", image["type"])
            self.assertEqual("base64", image["source"]["type"])
            self.assertEqual("image/png", image["source"]["media_type"])
            self.assertEqual(_PNG, base64.b64decode(image["source"]["data"]))

    def test_invalid_file_content_is_rejected_even_with_image_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fake.png"
            source.write_text("not an image", encoding="utf-8")

            with self.assertRaisesRegex(ImageInputError, "JPEG、PNG、GIF 或 WebP"):
                build_image_user_content(str(source), "", project_root=root)

    def test_history_estimates_image_tokens_without_counting_binary_payload(self):
        history = ConversationHistory()
        history.add_user_message([
            {"type": "text", "text": "分析图片"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 100_000},
            },
        ])

        estimated = history.estimated_token_count()

        self.assertGreaterEqual(estimated, IMAGE_TOKEN_ESTIMATE)
        self.assertLess(estimated, 2_000)

    def test_user_content_description_hides_attachment_path(self):
        content = [
            {"type": "text", "text": "看下这个错误"},
            {
                "type": "image_file",
                "image_file": {
                    "path": "/private/secret/abc.png",
                    "media_type": "image/png",
                    "detail": "auto",
                    "size": 10,
                },
            },
        ]

        rendered = describe_user_content(content)

        self.assertIn("abc.png", rendered)
        self.assertNotIn("/private/secret", rendered)


class NativeImagePickerTests(unittest.IsolatedAsyncioTestCase):
    async def test_macos_picker_returns_selected_posix_path(self):
        process = type("PickerProcess", (), {
            "returncode": 0,
            "communicate": AsyncMock(return_value=(b"/tmp/screen.png\n", b"")),
        })()
        with (
            patch("tinyCode.multimodal.sys.platform", "darwin"),
            patch(
                "tinyCode.multimodal.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=process),
            ) as create,
        ):
            selected = await select_local_image()

        self.assertEqual("/tmp/screen.png", selected)
        self.assertEqual("/usr/bin/osascript", create.await_args.args[0])

    async def test_macos_clipboard_image_is_persisted_as_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = type("ClipboardProcess", (), {
                "returncode": 0,
                "communicate": AsyncMock(return_value=(b"png\n", b"")),
            })()

            async def create_process(*args, **kwargs):
                del kwargs
                Path(args[-1]).write_bytes(_PNG)
                return process

            with (
                patch("tinyCode.multimodal.sys.platform", "darwin"),
                patch(
                    "tinyCode.multimodal.asyncio.create_subprocess_exec",
                    new=AsyncMock(side_effect=create_process),
                ),
            ):
                selected = await paste_clipboard_image(root)

            attachment = Path(selected or "")
            self.assertTrue(attachment.is_file())
            self.assertEqual(_PNG, attachment.read_bytes())
            self.assertEqual(
                (root / ".tinyCode" / "attachments").resolve(),
                attachment.parent,
            )
            self.assertFalse(any(
                path.name.startswith(".clipboard-")
                for path in attachment.parent.iterdir()
            ))
