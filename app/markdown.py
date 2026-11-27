"""把验收清单用的那一小撮 Markdown 渲染成 HTML（不引第三方库，不联网）。

只用阿配 ``render_acceptance_markdown`` 真正会产出的语法：
标题（``#``~``###``）、表格（``|``）、引用（``> ``）、有序/无序列表、
行内代码（反引号）与粗体（``**``）。其余文本按段落输出。

安全：先整体 HTML 转义，再把受控的标记换成标签；用户输入（触发词、路径）
只会出现在文本或行内代码里，不会变成标签。
"""

from __future__ import annotations

import re
from html import escape

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
_ORDERED = re.compile(r"^\d+\.\s+(.*)$")
_TABLE_SEP = re.compile(r"^\|(\s*:?-{2,}:?\s*\|)+\s*$")


def _inline(text: str) -> str:
    """行内标记：先转义，再换 ``code`` 与 **bold**。"""
    out = escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return out


def _cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def render_markdown(text: str) -> str:
    """返回 HTML 片段（不含 ``<html>`` 外壳）。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    html: list[str] = []
    index = 0
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            html.append("<p>" + _inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush()
            index += 1
            continue

        heading = _HEADING.match(stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            html.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and _TABLE_SEP.match(
            lines[index + 1].strip()
        ):
            flush()
            header = _cells(stripped)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(_cells(lines[index]))
                index += 1
            out = ["<table><thead><tr>"]
            out.extend(f"<th>{_inline(cell)}</th>" for cell in header)
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>")
                out.extend(f"<td>{_inline(cell)}</td>" for cell in row)
                out.append("</tr>")
            out.append("</tbody></table>")
            html.append("".join(out))
            continue

        if stripped.startswith(">"):
            flush()
            quoted: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quoted.append(lines[index].strip()[1:].strip())
                index += 1
            html.append("<blockquote>" + _inline(" ".join(quoted)) + "</blockquote>")
            continue

        if stripped.startswith("- "):
            flush()
            items: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(lines[index].strip()[2:].strip())
                index += 1
            html.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
            continue

        ordered = _ORDERED.match(stripped)
        if ordered:
            flush()
            items = []
            while index < len(lines):
                match = _ORDERED.match(lines[index].strip())
                if not match:
                    break
                items.append(match.group(1).strip())
                index += 1
            html.append("<ol>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ol>")
            continue

        paragraph.append(stripped)
        index += 1

    flush()
    return "\n".join(html)


__all__ = ["render_markdown"]
