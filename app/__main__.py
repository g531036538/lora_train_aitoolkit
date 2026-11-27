"""``python -m app`` 的入口：本机 uvicorn 起在 127.0.0.1:7861。"""

from __future__ import annotations

import uvicorn

HOST = "127.0.0.1"
PORT = 7861


def main() -> None:
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False, log_level="info")


if __name__ == "__main__":
    main()
