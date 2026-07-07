from __future__ import annotations

import html
import re
from typing import Dict


def clean_html(html_content: str) -> str:
    if not html_content:
        return ""

    code_placeholders: Dict[str, str] = {}

    def stash_code_block(match: re.Match) -> str:
        placeholder = f"@@TL_CODE_BLOCK_{len(code_placeholders)}@@"
        code_placeholders[placeholder] = f"```\n{html.unescape(match.group(1))}\n```"
        return placeholder

    def stash_inline_code(match: re.Match) -> str:
        placeholder = f"@@TL_INLINE_CODE_{len(code_placeholders)}@@"
        code_placeholders[placeholder] = f"`{html.unescape(match.group(1))}`"
        return placeholder

    text = re.sub(
        r"<pre[^>]*><code[^>]*>(.*?)</code></pre>",
        stash_code_block,
        html_content,
        flags=re.DOTALL,
    )
    text = re.sub(r"<code>(.*?)</code>", stash_inline_code, text, flags=re.DOTALL)
    text = html.unescape(text)
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", text, flags=re.DOTALL)
    text = re.sub(r"<li>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<ul>|</ul>|<ol>|</ol>", "", text)
    text = re.sub(r"<p>(.*?)</p>", r"\1\n\n", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text, flags=re.DOTALL)
    text = re.sub(r"<em>(.*?)</em>", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    for placeholder, replacement in code_placeholders.items():
        text = text.replace(placeholder, replacement)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def assess_difficulty(body_html: str, answer_html: str | None) -> str:
    body = clean_html(body_html)
    answer = clean_html(answer_html or "")
    body_len = len(body)
    context_len = body_len + len(answer)
    code_blocks = body.count("```") // 2

    if body_len <= 1200 and context_len <= 3000 and code_blocks <= 1:
        return "easy"
    if body_len >= 5000 or context_len >= 10000 or code_blocks >= 4:
        return "hard"
    return "medium"
