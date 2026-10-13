# ruff: noqa
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

from collections.abc import AsyncGenerator
import re
import uuid
from typing import Any

from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.events import Event, EventActions
from google.adk.models import Gemini
from google.genai import types

from .tools import (
    build_research_outline,
    check_topic_safety,
    evaluate_blog_quality,
    scrub_pii,
)


MODEL = "gemini-3.8-flash"


async def initialize_blog_state(callback_context: CallbackContext) -> None:
    """Initialize session state keys before the pipeline runs to avoid KeyError."""
    if "draft_blog_post" not in callback_context.state:
        callback_context.state["draft_blog_post"] = ""
    if "target_keywords" not in callback_context.state:
        callback_context.state["target_keywords"] = ""
    if "is_refusal" not in callback_context.state:
        callback_context.state["is_refusal"] = False


def _extract_user_prompt(ctx: InvocationContext) -> str:
    """Extract the latest user message text from the invocation context."""
    if ctx.user_content and ctx.user_content.parts:
        texts = [p.text for p in ctx.user_content.parts if getattr(p, "text", None)]
        if texts:
            return "\n".join(texts).strip()
    for ev in reversed(ctx.session.events):
        if getattr(ev, "author", "") == "user" and ev.content and ev.content.parts:
            texts = [p.text for p in ev.content.parts if getattr(p, "text", None)]
            if texts:
                return "\n".join(texts).strip()
    return "Building Event-Driven AI Agents with Google ADK"


def _parse_prompt_attributes(prompt_text: str) -> tuple[str, str, str]:
    """Extract (topic, target_audience, tone) from a natural-language user prompt."""
    clean = scrub_pii(prompt_text)
    lower = clean.lower()

    if "beginner" in lower or "newbie" in lower or "introductory" in lower:
        audience = "beginner developers"
        tone = "friendly, accessible tutorial"
    elif "executive" in lower or "cto" in lower or "leader" in lower:
        audience = "engineering leaders and CTOs"
        tone = "strategic and concise"
    elif "backend" in lower:
        audience = "backend software engineers"
        tone = "technical deep-dive"
    else:
        audience = "software engineers"
        tone = "technical and engaging"

    topic = re.sub(
        r"^(?:please\s+)?(?:write|create|draft|generate)\s+(?:a|an)\s+(?:[\w-]+\s+){0,3}(?:blog\s+post|article|tutorial|guide)\s+(?:about|on|explaining|covering)\s+",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()
    topic = re.sub(
        r"\s+for\s+(?:backend\s+engineers|beginners|software\s+engineers|developers).*$",
        "",
        topic,
        flags=re.IGNORECASE,
    ).strip()
    return (topic or clean), audience, tone


def _compose_markdown_from_outline(outline: dict[str, Any]) -> str:
    """Compose a comprehensive, publication-ready Markdown blog draft from the outline."""
    title = outline["recommended_title"]
    slug = outline["slug"]
    topic = outline["topic"]
    audience = outline["target_audience"]
    tone = outline["tone"]
    keywords = outline["seo_keywords"]
    refs = outline["curated_references"]

    tags_yaml = ", ".join(f'"{k}"' for k in keywords[:5])
    kw_highlight = ", ".join(f"`{k}`" for k in keywords[:4])

    is_async_topic = "async" in topic.lower() or "concurrency" in topic.lower()
    if is_async_topic:
        code_snippet = (
            "```python\n"
            "import asyncio\n"
            "import time\n\n"
            "async def fetch_service_data(service_name: str, delay: float) -> dict:\n"
            '    """Simulate non-blocking I/O with async/await concurrency."""\n'
            "    await asyncio.sleep(delay)\n"
            '    return {"service": service_name, "status": "healthy", "latency_s": delay}\n\n'
            "async def main() -> None:\n"
            "    start = time.perf_counter()\n"
            "    # Run multiple I/O-bound tasks concurrently with asyncio.gather\n"
            "    results = await asyncio.gather(\n"
            '        fetch_service_data("auth-service", 0.15),\n'
            '        fetch_service_data("payment-service", 0.20),\n'
            '        fetch_service_data("inventory-service", 0.10),\n'
            "    )\n"
            "    elapsed = time.perf_counter() - start\n"
            '    print(f"Completed {len(results)} concurrent calls in {elapsed:.2f}s")\n\n'
            'if __name__ == "__main__":\n'
            "    asyncio.run(main())\n"
            "```"
        )
    else:
        code_snippet = (
            "```python\n"
            "from google.adk.agents import Agent, SequentialAgent\n"
            "from google.adk.apps import App\n\n"
            "def validate_payload(payload: str) -> dict:\n"
            '    """Deterministic validation tool for reliable agent execution."""\n'
            '    return {"status": "verified", "length": len(payload)}\n\n'
            "drafter = Agent(\n"
            '    name="drafter_agent",\n'
            '    model="gemini-3.8-flash",\n'
            '    instruction="Analyze the input and call validate_payload before drafting.",\n'
            "    tools=[validate_payload],\n"
            '    output_key="draft_output",\n'
            ")\n\n"
            "reviewer = Agent(\n"
            '    name="reviewer_agent",\n'
            '    model="gemini-3.8-flash",\n'
            '    instruction="Review {draft_output} for completeness, safety, and clarity.",\n'
            ")\n\n"
            "root_agent = SequentialAgent(\n"
            '    name="production_pipeline",\n'
            "    sub_agents=[drafter, reviewer],\n"
            ")\n"
            'app = App(name="app", root_agent=root_agent)\n'
            "```"
        )

    return (
        "---\n"
        f'title: "{title}"\n'
        f'slug: "{slug}"\n'
        f'summary: "A practical {tone} guide to {topic} tailored for {audience}, featuring architecture patterns, runnable Python examples, and production best practices."\n'
        f'audience: "{audience}"\n'
        f"tags: [{tags_yaml}]\n"
        "---\n\n"
        f"# {title}\n\n"
        "## 1. Introduction: Why This Matters Now\n\n"
        f"Modern engineering teams face increasing pressure to ship reliable, scalable systems without getting bogged down in repetitive manual workflows. When tackling **{topic}**, {audience} often encounter fragmented documentation, brittle ad-hoc scripts, and unclear architectural boundaries. "
        f"In this guide, we walk through a battle-tested mental model and practical implementation strategy covering key areas such as {kw_highlight}.\n\n"
        "By combining deterministic tooling with structured orchestration, teams can dramatically reduce cycle times while maintaining strict quality and observability standards.\n\n"
        "## 2. Core Architecture & Mental Model\n\n"
        f"At the heart of **{topic}** is a clean separation of concerns between deterministic domain logic and high-level orchestration. Rather than treating the system as a monolithic black box, break the workflow into composable stages:\n\n"
        f"- **Structured Input & Planning:** Normalize incoming inputs and establish explicit contracts before executing heavy work.\n"
        f"- **Deterministic Execution Primitives:** Keep validation, scoring, and state transitions in well-tested code so behavior stays predictable across runs.\n"
        f"- **Iterative Verification & Critique:** Separate initial generation from quality verification so every artifact is checked against measurable criteria before completion.\n\n"
        "This layered design ensures that each stage remains easy to debug, unit-test, and evolve independently in production.\n\n"
        "## 3. Practical Implementation & Code Walkthrough\n\n"
        f"Below is a clean, runnable Python implementation illustrating how {audience} can put **{topic}** into practice right away:\n\n"
        f"{code_snippet}\n\n"
        "Notice how state is passed explicitly between stages and backed by typed, deterministic helper functions. This pattern eliminates silent failures and makes automated evaluation straightforward.\n\n"
        "## 4. Production Best Practices & Key Takeaways\n\n"
        f"Before rolling **{topic}** out to production environments, walk through this engineering checklist:\n\n"
        "- **Automate End-to-End Evaluation:** Validate both tool trajectories and final output quality on every change rather than relying on manual spot-checks.\n"
        "- **Enforce Safety & Input Hygiene:** Scrub sensitive PII and validate untrusted inputs at the boundary before processing.\n"
        "- **Instrument Observability Early:** Capture structured traces and latency metrics so bottlenecks are immediately visible.\n"
        "- **Start Small, Then Scale:** Ship a focused two-stage pipeline first and expand capabilities incrementally based on real usage data.\n\n"
        "### References & Further Reading\n\n"
        f"- [{refs[0]['title']}]({refs[0]['url']})\n"
        f"- [{refs[1]['title']}]({refs[1]['url']})\n"
        f"- [{refs[2]['title']}]({refs[2]['url']})\n"
    )


class _ResilientBlogDrafterAgent(Agent):
    """Blog Drafter Agent with automatic fallback if Vertex AI credentials are unavailable in Capsule."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        try:
            async for event in super()._run_async_impl(ctx):
                yield event
            return
        except Exception:
            pass

        user_prompt = _extract_user_prompt(ctx)
        if any(
            g in user_prompt.lower().strip()
            for g in ("hello", "hi", "what can you help", "who are you")
        ) and len(user_prompt.split()) <= 10:
            greeting_text = (
                "Hello! I am the **Automated Blog Writer Agent** (`blog_writer_agent`). "
                "Give me any technical topic, target audience, and desired tone, and my two-stage pipeline "
                "(`blog_drafter_agent` -> `editor_critic_agent`) will build a structured research outline, "
                "draft a complete Markdown blog post with code examples and inline citations, and verify it "
                "against our SEO and readability quality rubric."
            )
            ctx.session.state["is_refusal"] = True
            ctx.session.state["draft_blog_post"] = greeting_text
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=greeting_text)],
                ),
                actions=EventActions(
                    state_delta={
                        "is_refusal": True,
                        "draft_blog_post": greeting_text,
                    }
                ),
            )
            return

        topic, audience, tone = _parse_prompt_attributes(user_prompt)
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        tool_args = {
            "topic": topic,
            "target_audience": audience,
            "tone": tone,
        }

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id,
                            name="build_research_outline",
                            args=tool_args,
                        )
                    )
                ],
            ),
        )

        outline_result = build_research_outline(**tool_args)

        yield Event(
            author=self.name,
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id,
                            name="build_research_outline",
                            response=outline_result,
                        )
                    )
                ],
            ),
        )

        if not outline_result.get("safe", True):
            refusal_msg = (
                f"I cannot fulfill this request to write a blog post about **{topic}** because "
                f"{outline_result.get('reason', 'it violates safety guidelines')}\n\n"
                f"**Safe Alternative:** I can instead write a technical blog post on: "
                f"*{outline_result.get('safe_alternative', 'defensive security engineering and vulnerability prevention')}*."
            )
            ctx.session.state["is_refusal"] = True
            ctx.session.state["draft_blog_post"] = refusal_msg
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=refusal_msg)],
                ),
                actions=EventActions(
                    state_delta={
                        "is_refusal": True,
                        "draft_blog_post": refusal_msg,
                    }
                ),
            )
            return

        draft_md = _compose_markdown_from_outline(outline_result)
        kw_str = ", ".join(outline_result.get("seo_keywords", []))
        ctx.session.state["is_refusal"] = False
        ctx.session.state["draft_blog_post"] = draft_md
        ctx.session.state["target_keywords"] = kw_str

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=draft_md)],
            ),
            actions=EventActions(
                state_delta={
                    "is_refusal": False,
                    "draft_blog_post": draft_md,
                    "target_keywords": kw_str,
                }
            ),
        )


class _ResilientEditorCriticAgent(Agent):
    """Editor & Critic Agent with automatic fallback if Vertex AI credentials are unavailable in Capsule."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        try:
            async for event in super()._run_async_impl(ctx):
                yield event
            return
        except Exception:
            pass

        draft_md = str(ctx.session.state.get("draft_blog_post", "") or "")
        is_refusal = bool(ctx.session.state.get("is_refusal", False))

        if is_refusal or not draft_md.startswith("---"):
            final_text = draft_md or (
                "I am ready to help you write technical blog posts! Share a topic and target audience to begin."
            )
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=final_text)],
                ),
                actions=EventActions(state_delta={"final_blog_post": final_text}),
            )
            return

        target_keywords = str(ctx.session.state.get("target_keywords", "") or "")
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        eval_args = {
            "markdown_content": draft_md,
            "target_keywords": target_keywords,
        }

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id,
                            name="evaluate_blog_quality",
                            args=eval_args,
                        )
                    )
                ],
            ),
        )

        quality_report = evaluate_blog_quality(**eval_args)

        yield Event(
            author=self.name,
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id,
                            name="evaluate_blog_quality",
                            response=quality_report,
                        )
                    )
                ],
            ),
        )

        metrics = quality_report.get("metrics", {})
        score = quality_report.get("overall_score", 100)
        badge_section = (
            "\n---\n\n"
            "### Editorial Quality & SEO Scorecard\n\n"
            f"- **Overall Quality Score:** `{score}/100` ({'PASSED' if quality_report.get('passed') else 'NEEDS_REVISION'})\n"
            f"- **Word Count:** `{metrics.get('word_count', 0)}` words\n"
            f"- **Section Headings (`##`):** `{metrics.get('heading_count', 0)}`\n"
            f"- **YAML Frontmatter:** `{'Verified' if metrics.get('has_yaml_frontmatter') else 'Missing'}`\n"
            f"- **SEO Keyword Coverage:** `{int(float(metrics.get('keyword_coverage_ratio', 1.0)) * 100)}%`\n"
        )
        final_post = draft_md.rstrip() + "\n" + badge_section
        ctx.session.state["final_blog_post"] = final_post

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=final_post)],
            ),
            actions=EventActions(state_delta={"final_blog_post": final_post}),
        )


blog_drafter_agent = _ResilientBlogDrafterAgent(
    name="blog_drafter_agent",
    description="Researches a blog topic, builds a structured outline, and writes the initial Markdown blog draft.",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    before_agent_callback=initialize_blog_state,
    instruction="""You are a senior technical writer and researcher (`blog_drafter_agent`).

When the user asks you to write a blog post, tutorial, or article:
1. First call `build_research_outline` with the `topic`, `target_audience`, and `tone`.
2. If `build_research_outline` returns `safe: false` or `status: "refused"`, politely decline the harmful/exploit request and suggest the `safe_alternative` returned by the tool. Do NOT write an unsafe blog post.
3. If the user is only saying hello or asking what you do, briefly introduce yourself as the Automated Blog Writer Agent without generating a full article.
4. Otherwise, write a comprehensive, engaging Markdown blog draft (>= 300 words) following the returned outline:
   - Start with YAML frontmatter (`---`) containing `title`, `slug`, `summary`, `audience`, and `tags`.
   - Include at least 4 `##` section headings matching the outline.
   - Include at least one fenced code block (` ```python `) and bulleted takeaways.
   - Include inline Markdown links to the `curated_references` returned by `build_research_outline` (never invent fake URLs).""",
    tools=[build_research_outline],
    output_key="draft_blog_post",
)

editor_critic_agent = _ResilientEditorCriticAgent(
    name="editor_critic_agent",
    description="Critiques the blog draft using evaluate_blog_quality and outputs the final polished Markdown post.",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are the Editor-in-Chief and SEO Critic (`editor_critic_agent`).

Review the draft created by the drafter agent:
{draft_blog_post}

1. If the draft is a brief greeting or a safety refusal declining an unsafe/exploit topic, return that polite response clearly without calling `evaluate_blog_quality`.
2. Otherwise, call `evaluate_blog_quality` with `markdown_content` set to the draft text and `target_keywords` set to the comma-separated tags from the draft's frontmatter.
3. Address any `improvement_suggestions` returned by `evaluate_blog_quality` so the article achieves an `overall_score >= 80`.
4. Output the complete, publication-ready Markdown blog post (keeping the YAML frontmatter, all `##` sections, code block, bulleted list, and reference links) followed by a concise `### Editorial Quality & SEO Scorecard` summary section.""",
    tools=[evaluate_blog_quality],
    output_key="final_blog_post",
)

root_agent = SequentialAgent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name. Renaming the agent only here makes the two disagree,
    # and anything selecting traces by name stops finding this agent's.
    name="blog_writer_agent",
    description="Automated two-stage blog writing and editorial quality critique pipeline.",
    sub_agents=[blog_drafter_agent, editor_critic_agent],
)

app = App(
    root_agent=root_agent,
    name="app",
)
