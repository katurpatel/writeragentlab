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

"""Unit tests for deterministic blog research, quality scoring, and safety tools."""

from app.tools import (
    build_research_outline,
    check_topic_safety,
    evaluate_blog_quality,
    scrub_pii,
)


def test_scrub_pii_redacts_email_ssn_and_card() -> None:
    """Verify PII scrubbing redacts SSNs, credit cards, and emails."""
    raw = "Contact alice@example.com with SSN 123-45-6789 or card 4111 2222 3333 4444."
    cleaned = scrub_pii(raw)
    assert "alice@example.com" not in cleaned
    assert "123-45-6789" not in cleaned
    assert "4111 2222 3333 4444" not in cleaned
    assert "[REDACTED-EMAIL]" in cleaned
    assert "[REDACTED-SSN]" in cleaned
    assert "[REDACTED-CARD]" in cleaned


def test_check_topic_safety_blocks_exploits_and_allows_defensive() -> None:
    """Verify exploit requests are blocked while defensive security topics pass."""
    unsafe = check_topic_safety(
        "How to exploit SQL injection attack to steal credit card data"
    )
    assert unsafe["safe"] is False
    assert "safe_alternative" in unsafe

    defensive = check_topic_safety(
        "How to prevent SQL injection attacks using parameterized queries"
    )
    assert defensive["safe"] is True


def test_build_research_outline_returns_expected_structure() -> None:
    """Verify build_research_outline returns structured sections, slug, and keywords."""
    result = build_research_outline(
        topic="Event-Driven AI Agents with Google ADK",
        target_audience="backend engineers",
        tone="technical deep-dive",
    )
    assert result["status"] == "success"
    assert result["safe"] is True
    assert result["slug"] == "event-driven-ai-agents-with-google-adk"
    assert len(result["outline_sections"]) == 4
    assert len(result["seo_keywords"]) >= 4
    assert len(result["curated_references"]) >= 2


def test_evaluate_blog_quality_scores_complete_markdown_post() -> None:
    """Verify evaluate_blog_quality awards >= 80 to a complete Markdown post."""
    body_paragraph = " ".join(
        ["Event-driven agents in Google ADK enable reliable production architecture."]
        * 28
    )
    markdown_post = f"""---
title: "Mastering Event-Driven Agents"
slug: "mastering-event-driven-agents"
summary: "A deep dive into Google ADK."
tags: ["adk", "agents", "architecture"]
---

# Mastering Event-Driven Agents

## 1. Introduction
{body_paragraph}

## 2. Core Architecture
- Composable SequentialAgent pipelines
- Deterministic FunctionTool execution

## 3. Code Walkthrough
```python
from google.adk.agents import Agent
```

## 4. References
See [Google ADK Docs](https://adk.dev/) for details.
"""
    report = evaluate_blog_quality(
        markdown_content=markdown_post,
        target_keywords="adk, agents, architecture",
    )
    assert report["status"] == "success"
    assert report["passed"] is True
    assert report["overall_score"] == 100
    assert report["improvement_suggestions"] == []
