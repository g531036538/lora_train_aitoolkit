#!/usr/bin/env python3
"""把本仓库的目标分支拉到本地目录，用于开发与跑测试。

我们的环境里 git/curl 的 schannel TLS 坏了，但 Python 能出网，
所以用 tarball 下载代替 git clone。

用法:
    python tools/fetch_repo.py --dest .                 # 拉到当前目录
    python tools/fetch_repo.py --dest ./work            # 拉到子目录
    python tools/fetch_repo.py --dest . --ref feat/xxx   # 拉某个分支
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import urllib.request

DEFAULT_REPO = "g531036538/lora_train_aitoolkit"


def fetch(repo: str, ref: str, dest: str) -> None:
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{ref}"
    req = urllib.request.Request(url, headers={"User-Agent": "loraforge-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = tf.getmembers()
        if not members:
            raise SystemExit("empty archive")
        prefix = members[0].name.split("/")[0] + "/"
        for m in members:
            if not m.name.startswith(prefix):
                continue
            rel = m.name[len(prefix) :]
            if not rel:
                continue
            m.name = rel
            try:
                tf.extract(m, dest, filter="data")  # py3.12+
            except TypeError:
                tf.extract(m, dest)
    print(f"fetched {repo}@{ref} -> {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=".")
    ap.add_argument("--ref", default="main")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    a = ap.parse_args()
    fetch(a.repo, a.ref, a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
