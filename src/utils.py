from pathlib import Path


def load_text_file(file_path: Path) -> str:
    with file_path.open(mode="r", encoding="utf-8") as f:
        return f.read()
