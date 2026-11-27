"""验收清单 Markdown 渲染器的小单测：表格/标题/行内标记 + 注入不成立。"""

from __future__ import annotations

from app.markdown import render_markdown


def test_headings_table_and_inline_markup() -> None:
    html = render_markdown(
        "\n".join(
            [
                "# 标题",
                "",
                "正文里 `code` 与 **bold**。",
                "",
                "| 参数 | 值 |",
                "| --- | --- |",
                "| 步数 | 25 |",
                "",
                "> 引用一句",
                "",
                "- 一条",
                "- 两条",
                "",
                "1. 第一",
                "2. 第二",
            ]
        )
    )

    assert "<h1>标题</h1>" in html
    assert "<code>code</code>" in html
    assert "<strong>bold</strong>" in html
    assert "<table><thead><tr><th>参数</th><th>值</th></tr></thead>" in html
    assert "<td>步数</td><td>25</td>" in html
    assert "<blockquote>引用一句</blockquote>" in html
    assert "<ul><li>一条</li><li>两条</li></ul>" in html
    assert "<ol><li>第一</li><li>第二</li></ol>" in html


def test_html_is_escaped_before_markup() -> None:
    html = render_markdown("触发词 <script>alert(1)</script> 与 `a<b>`")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<code>a&lt;b&gt;</code>" in html


def test_lf_input_has_no_stray_paragraph_breaks() -> None:
    html = render_markdown("第一段\n第二行\n\n第二段\n")
    assert html == "<p>第一段 第二行</p>\n<p>第二段</p>"
