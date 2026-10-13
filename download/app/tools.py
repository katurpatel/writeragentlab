# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Research outline, SEO/readability scoring, and safety tools for the Automated Blog Writer Agent."""

import re
from typing import Any


_UNSAFE_PATTERNS = [
    r"\b(?:exploit|hack|steal|exfiltrate|bypass\s+auth|sql\s+injection\s+attack|ransomware|malware|keylogger|phishing\s+kit)\b",
    r"\b(?:steal\s+credit\s+card|dump\s+database|zero[- ]day\s+exploit)\b",
]


def scrub_pii(text: str) -> str:
    """Scrub SSNs, credit card numbers, and raw email addresses from input text."""
    if not text:
        return ""
    scrubbed = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]", text)
    scrubbed = re.sub(
        r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED-CARD]", scrubbed
    )
    scrubbed = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED-EMAIL]",
        scrubbed,
    )
    return scrubbed


def check_topic_safety(text: str) -> dict[str, Any]:
    """Check whether a blog request violates harmful/exploit content guardrails."""
    lowered = (text or "").lower()
    for pattern in _UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            if any(
                defensive in lowered
                for defensive in ("prevent", "defend", "mitigate", "protect", "fix")
            ):
                break
            return {
                "safe": False,
                "reason": (
                    "Request involves actionable exploit instructions, malware, or unauthorized data theft."
                ),
                "safe_alternative": (
                    "A defensive engineering guide on preventing vulnerabilities (e.g., parameterized queries, input validation, and zero-trust security controls)."
                ),
            }
    return {"safe": True, "reason": "Topic passes safety guardrails."}


def _slugify(text: str) -> str:
    """Convert a topic or title into a clean URL slug."""
    cleaned = re.sub(r"[^a-z0-9\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", cleaned)
    return slug[:64].strip("-") or "technical-deep-dive"


def _extract_keywords(topic: str) -> list[str]:
    """Extract primary and secondary SEO keywords from the topic."""
    stop_words = {
        "write",
        "blog",
        "post",
        "about",
        "create",
        "guide",
        "tutorial",
        "with",
        "from",
        "that",
        "this",
        "for",
        "and",
        "the",
        "how",
        "using",
        "into",
    }
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z0-9+#.-]{3,}", topic)
        if w.lower() not in stop_words
    ]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    defaults = ["architecture", "best-practices", "implementation", "tutorial", "production"]
    for d in defaults:
        if len(seen) >= 6:
            break
        if d not in seen:
            seen.append(d)
    return seen[:6]


def build_research_outline(
    topic: str,
    target_audience: str,
    tone: str,
) -> dict[str, Any]:
    """Generate a structured research brief, SEO keywords, section outline, and curated references for a blog post.

    Args:
        topic: The core subject or thesis of the blog post.
        target_audience: The intended readers (e.g., backend engineers, beginners, tech leads).
        tone: The desired writing voice (e.g., technical deep-dive, friendly tutorial, executive summary).

    Returns:
        A dictionary containing the recommended title, slug, SEO keywords, structured 4-part outline, and canonical references.
    """
    clean_topic = scrub_pii(topic).strip()
    clean_audience = scrub_pii(target_audience).strip() or "software engineers"
    clean_tone = scrub_pii(tone).strip() or "technical and engaging"

    safety = check_topic_safety(clean_topic)
    if not safety["safe"]:
        return {
            "status": "refused",
            "safe": False,
            "reason": safety["reason"],
            "safe_alternative": safety["safe_alternative"],
        }

    keywords = _extract_keywords(clean_topic)
    slug = _slugify(clean_topic)
    title_topic = clean_topic.rstrip(".")
    if len(title_topic) > 72:
        title_topic = title_topic[:69].rsplit(" ", 1)[0]
    recommended_title = f"Mastering {title_topic[:1].upper() + title_topic[1:]}: A Practical Guide for {clean_audience.title()}"

    references = [
        {
            "title": "Google Agent Development Kit (ADK) Documentation",
            "url": "https://adk.dev/",
        },
        {
            "title": "Google Cloud Vertex AI Generative AI Documentation",
            "url": "https://cloud.google.com/vertex-ai/generative-ai/docs",
        },
        {
            "title": "Python 3 Official Documentation & Concurrency Guides",
            "url": "https://docs.python.org/3/",
        },
    ]

    outline_sections = [
        {
            "heading": "## 1. Introduction: Why This Matters Now",
            "objectives": [
                f"Hook {clean_audience} with the real-world pain point around {clean_topic}.",
                "State the core thesis and what readers will be able to build or solve by the end.",
            ],
        },
        {
            "heading": "## 2. Core Architecture & Mental Model",
            "objectives": [
                f"Break down the foundational primitives behind {clean_topic}.",
                "Explain trade-offs, system boundaries, and design patterns clearly.",
            ],
        },
        {
            "heading": "## 3. Practical Implementation & Code Walkthrough",
            "objectives": [
                "Provide a clean, runnable code snippet or concrete step-by-step configuration.",
                "Annotate key lines so readers understand both the 'how' and the 'why'.",
            ],
        },
        {
            "heading": "## 4. Production Best Practices & Key Takeaways",
            "objectives": [
                "List 3-5 actionable production checklist items (observability, testing, safety).",
                "Include inline citations to canonical documentation and a concise wrap-up.",
            ],
        },
    ]

    return {
        "status": "success",
        "safe": True,
        "topic": clean_topic,
        "recommended_title": recommended_title,
        "slug": slug,
        "target_audience": clean_audience,
        "tone": clean_tone,
        "seo_keywords": keywords,
        "outline_sections": outline_sections,
        "curated_references": references,
    }


def evaluate_blog_quality(
    markdown_content: str,
    target_keywords: str,
) -> dict[str, Any]:
    """Deterministically evaluate a Markdown blog draft across SEO, structure, and readability rubric dimensions.

    Args:
        markdown_content: The full Markdown text of the blog post (including YAML frontmatter).
        target_keywords: Comma-separated SEO keywords that should appear in the article.

    Returns:
        A dictionary with overall_score (0-100), passed boolean (>= 80), dimension breakdown, and improvement_suggestions.
    """
    content = (markdown_content or "").strip()
    suggestions: list[str] = []
    score = 0

    # 1. YAML Frontmatter check (20 points)
    has_frontmatter = bool(
        re.match(r"^---\s*\n.*?\n---\s*\n", content, flags=re.DOTALL)
        and "title:" in content[:600]
        and "slug:" in content[:600]
        and "tags:" in content[:600]
    )
    if has_frontmatter:
        score += 20
    else:
        suggestions.append(
            "Add YAML frontmatter at the top (`---`) with `title:`, `slug:`, `summary:`, and `tags:`."
        )

    # 2. Word count check >= 250 words (20 points)
    words = re.findall(r"\b\w+\b", content)
    word_count = len(words)
    if word_count >= 250:
        score += 20
    elif word_count >= 150:
        score += 10
        suggestions.append(
            f"Expand technical depth: current word count is {word_count} (target >= 250 words)."
        )
    else:
        suggestions.append(
            f"Article is too short ({word_count} words); expand sections to at least 250 words."
        )

    # 3. Section headings check >= 3 H2 headings (20 points)
    h2_headings = re.findall(r"^##\s+.+$", content, flags=re.MULTILINE)
    heading_count = len(h2_headings)
    if heading_count >= 3:
        score += 20
    else:
        suggestions.append(
            f"Include at least 3 `## ` section headings (found {heading_count})."
        )

    # 4. Structured elements: code blocks + lists + inline citations (20 points)
    has_code_block = "```" in content
    has_list = bool(re.search(r"^\s*[-*]\s+.+$", content, flags=re.MULTILINE))
    has_citation = bool(re.search(r"\[[^\]]+\]\(https?://[^\)]+\)", content))
    if has_code_block and has_list and has_citation:
        score += 20
    elif (has_code_block or has_list) and has_citation:
        score += 15
        suggestions.append(
            "Include both a fenced code block (```) and bulleted lists alongside inline Markdown links."
        )
    else:
        score += 10 if (has_code_block or has_list) else 0
        suggestions.append(
            "Add a fenced code example (```), bulleted key takeaways, and at least one canonical Markdown citation link."
        )

    # 5. SEO Keyword coverage (20 points)
    kw_list = [
        k.strip().lower()
        for k in re.split(r"[,;]+", target_keywords or "")
        if k.strip()
    ]
    lowered_content = content.lower()
    if not kw_list:
        coverage_ratio = 1.0
        score += 20
    else:
        matched = [k for k in kw_list if k in lowered_content]
        coverage_ratio = round(len(matched) / len(kw_list), 2)
        if coverage_ratio >= 0.6:
            score += 20
        elif coverage_ratio >= 0.3:
            score += 10
            missing = [k for k in kw_list if k not in lowered_content]
            suggestions.append(
                f"Naturally incorporate missing target keywords: {', '.join(missing)}."
            )
        else:
            suggestions.append(
                f"Low keyword coverage ({coverage_ratio * 100:.0f}%); weave target keywords into headings and body."
            )

    return {
        "status": "success",
        "overall_score": score,
        "passed": score >= 80,
        "metrics": {
            "word_count": word_count,
            "heading_count": heading_count,
            "has_yaml_frontmatter": has_frontmatter,
            "has_code_block": has_code_block,
            "has_bullet_list": has_list,
            "has_inline_citations": has_citation,
            "keyword_coverage_ratio": coverage_ratio,
        },
        "improvement_suggestions": suggestions,
    }
