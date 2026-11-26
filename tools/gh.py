#!/usr/bin/env python3
"""仓库协作 CLI。我们的环境里 git 的 schannel TLS 坏了，所以走 GitHub HTTP API。

需要环境变量 DSH_SECRET_GITHUB_TOKEN（细粒度 PAT：Contents / Issues / Pull requests 读写）。

用法:
    python tools/gh.py fetch --dest .                 # 拉 main 到本地目录（覆盖同名文件）
    python tools/gh.py fetch --dest . --ref feat/x
    python tools/gh.py get docs/ARCHITECTURE.md
    python tools/gh.py issues                          # 列 open issues
    python tools/gh.py issue-comment 3 --body-file note.md
    python tools/gh.py branch feat/caption-styler      # 从 main 建分支
    python tools/gh.py commit --branch feat/x --message "feat: ..." \
        --files loraforge/caption/base.py tests/test_caption_styler.py
    python tools/gh.py pr --head feat/x --title "..." --body-file pr.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request

OWNER = "g531036538"
REPO = "lora_train_aitoolkit"
API = "https://api.github.com"


def token() -> str:
    t = os.environ.get("DSH_SECRET_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not t:
        raise SystemExit(
            "缺少 DSH_SECRET_GITHUB_TOKEN。该仓库是私有的，所有读写都需要 PAT。"
        )
    return t


def _open(url: str, method: str = "GET", payload: dict | None = None, raw: bool = False):
    data = None
    headers = {
        "Authorization": f"Bearer {token()}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "loraforge-gh",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
            return body if raw else (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"HTTP {e.code} {method} {url}\n{detail}")


def api(path: str, method: str = "GET", payload: dict | None = None):
    return _open(f"{API}/repos/{OWNER}/{REPO}{path}", method, payload)


# ---------------------------------------------------------------- fetch


def cmd_fetch(a) -> None:
    url = f"https://codeload.github.com/{OWNER}/{REPO}/tar.gz/refs/heads/{a.ref}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token()}", "User-Agent": "loraforge-gh"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dest = os.path.abspath(a.dest)
    os.makedirs(dest, exist_ok=True)
    n = 0
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
                tf.extract(m, dest, filter="data")
            except TypeError:
                tf.extract(m, dest)
            if m.isfile():
                n += 1
    print(f"fetched {OWNER}/{REPO}@{a.ref} -> {dest} ({n} files)")


def cmd_get(a) -> None:
    body = api(f"/contents/{a.path}" + (f"?ref={a.ref}" if a.ref else ""))
    import base64

    data = base64.b64decode(body["content"])
    if a.out:
        with open(a.out, "wb") as f:
            f.write(data)
        print(f"wrote {a.out} ({len(data)} bytes)")
        return
    sys.stdout.write(data.decode("utf-8", "replace"))


# 主管拥有的共享契约文件。成员用 `sync` 一次性拉到最新，不要手抄。
CONTRACT_FILES = [
    "docs/ARCHITECTURE.md",
    "docs/WORKFLOW.md",
    "loraforge/models.py",
    "loraforge/caption/base.py",
    "loraforge/testing/__init__.py",
    "loraforge/testing/fakes.py",
    "tests/conftest.py",
    "pyproject.toml",
]


def cmd_sync(a) -> None:
    """把共享契约文件更新到 main 的最新版本。只碰这些文件，不碰你的代码。"""
    for path in CONTRACT_FILES:
        try:
            body = api(f"/contents/{path}")
        except SystemExit as e:
            print(f"skip {path}: {e}")
            continue
        import base64

        data = base64.b64decode(body["content"])
        cur = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                cur = f.read()
        if cur == data:
            print(f"ok   {path}")
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        print(f"sync {path} ({len(data)} bytes)")


# ---------------------------------------------------------------- issues


def cmd_issues(a) -> None:
    items = api(f"/issues?state={a.state}&per_page=50")
    for it in items:
        if "pull_request" in it:
            continue
        print(f"#{it['number']}\t{it['state']}\t{it['title']}")
        print(f"  {it['html_url']}")


def cmd_issue_comment(a) -> None:
    body = _read_body(a)
    r = api(f"/issues/{a.number}/comments", "POST", {"body": body})
    print(f"commented: {r['html_url']}")


def cmd_issue_create(a) -> None:
    body = _read_body(a)
    title = _read_text(a.title, getattr(a, "title_file", None))
    labels = [s for s in (a.labels or "").split(",") if s]
    payload = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    r = api("/issues", "POST", payload)
    print(f"created #{r['number']}: {r['html_url']}")


def _read_body(a) -> str:
    if getattr(a, "body_file", None):
        with open(a.body_file, "r", encoding="utf-8") as f:
            return f.read()
    return getattr(a, "body", None) or ""


# ---------------------------------------------------------------- git


def _head_sha(branch: str) -> str:
    try:
        return api(f"/git/ref/heads/{branch}")["object"]["sha"]
    except SystemExit:
        raise


def cmd_branch(a) -> None:
    base = _head_sha(a.base)
    try:
        api(f"/git/refs", "POST", {"ref": f"refs/heads/{a.name}", "sha": base})
        print(f"created branch {a.name} from {a.base} ({base[:8]})")
    except SystemExit as e:
        if "422" in str(e) or "already exists" in str(e):
            print(f"branch {a.name} already exists")
        else:
            raise


def _read_text(value, path) -> str:
    """标题等短文本也支持从文件读，理由同 _read_message。"""
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    if value is None:
        raise SystemExit("缺少文本参数（可用 --*-file 指定 UTF-8 文件）")
    return value


def _read_message(a) -> str:
    """提交信息优先从文件读。

    本机 shell 的 locale 不是 UTF-8，中文经命令行参数传到 Python 时会变成乱码，
    所以含中文的 message 一律写进文件再用 --message-file。
    """
    if getattr(a, "message_file", None):
        with open(a.message_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return a.message


def cmd_commit(a) -> None:
    message = _read_message(a)
    head = _head_sha(a.branch)
    base_tree = api(f"/git/commits/{head}")["tree"]["sha"]

    tree = []
    for path in a.files:
        local = os.path.join(a.root, path) if a.root else path
        with open(local, "rb") as f:
            content = f.read()
        blob = api("/git/blobs", "POST", {"content": content.decode("utf-8"), "encoding": "utf-8"})
        tree.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print(f"  blob {path} ({len(content)} bytes)")

    if not tree:
        raise SystemExit("no files given")

    new_tree = api("/git/trees", "POST", {"base_tree": base_tree, "tree": tree})
    commit = api(
        "/git/commits",
        "POST",
        {"message": message, "tree": new_tree["sha"], "parents": [head]},
    )
    api(f"/git/refs/heads/{a.branch}", "PATCH", {"sha": commit["sha"]})
    print(f"committed {commit['sha'][:8]} to {a.branch}")


def cmd_pr(a) -> None:
    body = _read_body(a)
    title = _read_text(a.title, getattr(a, "title_file", None))
    r = api(
        "/pulls",
        "POST",
        {"title": title, "head": a.head, "base": a.base, "body": body},
    )
    print(f"PR #{r['number']}: {r['html_url']}")


# ---------------------------------------------------------------- main


def main() -> int:
    try:  # 控制台 locale 不是 UTF-8，别让中文 print 直接把脚本打崩
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch")
    p.add_argument("--dest", default=".")
    p.add_argument("--ref", default="main")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("get")
    p.add_argument("path")
    p.add_argument("--out", default=None, help="写进文件，避开控制台编码问题")
    p.add_argument("--ref", default=None)
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("sync", help="更新共享契约文件到 main 最新")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("issues")
    p.add_argument("--state", default="open")
    p.set_defaults(func=cmd_issues)

    p = sub.add_parser("issue-comment")
    p.add_argument("number", type=int)
    p.add_argument("--body", default=None)
    p.add_argument("--body-file", default=None)
    p.set_defaults(func=cmd_issue_comment)

    p = sub.add_parser("issue-create")
    p.add_argument("--title", default=None)
    p.add_argument("--title-file", default=None)
    p.add_argument("--body", default=None)
    p.add_argument("--body-file", default=None)
    p.add_argument("--labels", default=None)
    p.set_defaults(func=cmd_issue_create)

    p = sub.add_parser("branch")
    p.add_argument("name")
    p.add_argument("--base", default="main")
    p.set_defaults(func=cmd_branch)

    p = sub.add_parser("commit")
    p.add_argument("--branch", required=True)
    p.add_argument("--message", default=None)
    p.add_argument("--message-file", default=None, help="UTF-8 文件，中文提交信息用这个")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--root", default=None)
    p.set_defaults(func=cmd_commit)

    p = sub.add_parser("pr")
    p.add_argument("--head", required=True)
    p.add_argument("--base", default="main")
    p.add_argument("--title", default=None)
    p.add_argument("--title-file", default=None)
    p.add_argument("--body", default=None)
    p.add_argument("--body-file", default=None)
    p.set_defaults(func=cmd_pr)

    a = ap.parse_args()
    if a.cmd == "commit" and not (a.message or a.message_file):
        raise SystemExit("commit 需要 --message 或 --message-file")
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
