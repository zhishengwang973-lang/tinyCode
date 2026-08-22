"""Auto-note categories and storage paths."""

from pathlib import Path

# User-level notes (stored in ~/.tinyCode/notes/)
USER_CATEGORIES = {
    "用户偏好": "user_preferences.md",
    "纠正反馈": "corrections.md",
}

# Project-level notes (stored in .tinyCode/notes/)
PROJECT_CATEGORIES = {
    "项目知识": "project_knowledge.md",
    "参考资料": "references.md",
}

NOTE_CATEGORY_DESCRIPTIONS = {
    "用户偏好": "用户的工作习惯、编码偏好、工具选择等",
    "纠正反馈": "用户纠正过的错误、用户不喜欢的做法",
    "项目知识": "项目技术栈、架构决策、关键文件位置等",
    "参考资料": "用户提到的文档链接、重要文章、参考资源",
}

# Prompt template for note updates
_NOTE_UPDATE_PROMPT = """\
你是一个笔记管理助手。请阅读当前笔记和最近的对话，更新一个指定分类的笔记。

**规则**：
- 目标分类：{category}
- 分类说明：{category_description}
- 只记录与目标分类相关的新信息，已有信息不要重复
- 如果某条新信息与已有内容冲突，更新旧内容并标注日期
- 保持简洁：每条信息一行，用 "- " 开头
- 保留已有笔记的结构和内容，只做增量修改
- 不要删除任何已有信息，除非它与新信息明确冲突

---

当前笔记：
{current_notes}

---

最近对话：
{recent_conversation}

---

只输出这个分类的完整笔记内容（从 "## {category}" 标题开始），不要输出其他分类，不要加额外说明。"""


def build_note_prompt(category: str, current_notes: str, recent_text: str) -> str:
    return _NOTE_UPDATE_PROMPT.format(
        category=category,
        category_description=NOTE_CATEGORY_DESCRIPTIONS.get(category, "用户指定的笔记分类"),
        current_notes=current_notes or "(暂无笔记)",
        recent_conversation=recent_text[:8000],
    )


def get_user_notes_dir() -> Path:
    return Path.home() / ".tinyCode" / "notes"


def get_project_notes_dir(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / ".tinyCode" / "notes"


def get_notes_file(category: str) -> str | None:
    """Return the filename for a category, or None if unknown."""
    all_cats = {**USER_CATEGORIES, **PROJECT_CATEGORIES}
    return all_cats.get(category)
