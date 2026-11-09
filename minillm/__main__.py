"""``python -m minillm`` 入口：等价于 ``python -m minillm.cli``。"""

from __future__ import annotations

from minillm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
