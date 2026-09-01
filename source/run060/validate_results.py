"""结果期I/O包装：保持冻结validate.py逻辑不变，只修复NumPy标量JSON序列化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import validate as frozen_validation


ORIGINAL_DUMPS = json.dumps


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ORIGINAL_DUMPS(safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def safe_dumps(payload: Any, *args: Any, **kwargs: Any) -> str:
    return ORIGINAL_DUMPS(safe(payload), *args, **kwargs)


def main() -> None:
    frozen_validation.write_json = safe_write_json
    frozen_validation.json.dumps = safe_dumps
    frozen_validation.main()


if __name__ == "__main__":
    main()
