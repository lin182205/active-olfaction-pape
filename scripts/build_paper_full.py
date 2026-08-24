"""按章节顺序重建完整论文 Markdown。"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS_DIR = ROOT / "chapters"
OUTPUT = ROOT / "paper_full.md"


def main() -> None:
    chapter_paths = sorted(CHAPTERS_DIR.glob("[0-9][0-9]_*.md"))
    if not chapter_paths:
        raise RuntimeError(f"未找到章节文件: {CHAPTERS_DIR}")

    sections = [path.read_text(encoding="utf-8").strip() for path in chapter_paths]
    OUTPUT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"已合并 {len(chapter_paths)} 个章节 -> {OUTPUT.name}")


if __name__ == "__main__":
    main()
