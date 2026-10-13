# Automated Blog Writer Agent (`blog-writer-agent`)

> **Build Agent Challenge Submission** — Built with [Google Agent Development Kit (ADK)](https://adk.dev/) and `google-agents-cli` (`v1.9.0`).

---

## 1. Problem & Solution Statement

- **Problem:** Writing well-researched, technically accurate, and SEO-optimized blog posts is too manual and time-intensive—planning outlines, synthesizing key facts, drafting runnable code examples, and editing for readability takes hours per article.
- **Solution:** I built an **Automated Blog Writer Agent** (`blog_writer_agent`) using a two-stage Google ADK `SequentialAgent` pipeline (`blog_drafter_agent` $\rightarrow$ `editor_critic_agent`) that transforms a topic, target audience, and desired tone into a publication-ready Markdown blog post complete with YAML frontmatter, runnable code examples, canonical inline citations, and a deterministic 100-point SEO & Readability Quality Scorecard.

---

## 2. Architecture

```mermaid
flowchart LR
    U["User Prompt\n(Topic, Audience, Tone)"] --> D["blog_drafter_agent\n(ADK Agent)"]
    D <--> T1["build_research_outline()\n• PII Scrubbing\n• Safety Guardrail\n• SEO Keywords & 4-Part Outline"]
    D -- "state['draft_blog_post']" --> E["editor_critic_agent\n(ADK Agent)"]
    E <--> T2["evaluate_blog_quality()\n• YAML Frontmatter (20pt)\n• Word Count >= 250 (20pt)\n• >= 3 H2 Headings (20pt)\n• Code + Lists + Links (20pt)\n• Keyword Coverage (20pt)"]
    E --> O["Final Markdown Blog Post\n+ Editorial Quality Scorecard"]
```

### Multi-Agent Pipeline & Tools

| Component | Type | Responsibility |
| :--- | :--- | :--- |
| **`blog_writer_agent`** (`root_agent`) | `SequentialAgent` | Coordinates the two-stage drafting and editorial critique workflow. |
| **`blog_drafter_agent`** | `Agent` (`gemini-3.8-flash`) | Calls `build_research_outline` to generate an SEO-targeted 4-section brief, enforces safety guardrails, and writes the initial Markdown draft into `state["draft_blog_post"]`. |
| **`editor_critic_agent`** | `Agent` (`gemini-3.8-flash`) | Calls `evaluate_blog_quality` to score the draft against a 5-dimension rubric (`0–100`), resolves any structural gaps, and appends an **Editorial Quality & SEO Scorecard**. |
| **`build_research_outline`** | `FunctionTool` (`app/tools.py`) | Scrubs PII (`scrub_pii`), screens for harmful/exploit topics (`check_topic_safety`), and returns a recommended title, URL slug, primary/secondary SEO keywords, 4-part outline, and canonical documentation references. |
| **`evaluate_blog_quality`** | `FunctionTool` (`app/tools.py`) | Deterministically scores a Markdown article (`0–100`, passing threshold $\ge 80$) across YAML frontmatter, word count ($\ge 250$), $\ge 3$ `##` headings, code/list/citation presence, and SEO keyword coverage. |

---

## 3. Project Structure

```text
blog-writer-agent/
├── .agents-cli-spec.md        # Approved Phase 0 Agent Specification
├── agents-cli-manifest.yaml   # agents-cli project manifest
├── app/
│   ├── agent.py               # SequentialAgent pipeline (blog_drafter_agent -> editor_critic_agent)
│   ├── tools.py               # Deterministic outline, SEO rubric, and PII/safety guardrail tools
│   ├── fast_api_app.py        # FastAPI + A2A server entrypoint
│   └── app_utils/             # Telemetry, typing, and A2A helpers
├── tests/
│   ├── eval/
│   │   ├── eval_config.yaml   # Evaluation metrics config (custom_response_quality + trajectory)
│   │   ├── response_quality.py# LLM-as-judge & rubric evaluator
│   │   └── datasets/
│   │       └── basic-dataset.json # Core technical blog, beginner tutorial, and safety refusal cases
│   ├── integration/           # Runner streaming & FastAPI server E2E tests
│   └── unit/                  # Deterministic unit tests for tools & safety guardrails
├── Dockerfile                 # Container image definition
└── pyproject.toml             # Project dependencies and configuration
```

---

## 4. Quick Start & Usage

### Prerequisites
- **Python 3.11+** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **`google-agents-cli`**: `uv tool install google-agents-cli`
- **Google Cloud Project** (Vertex AI enabled) or **Gemini API Key** (configured in `.env`)

### 1. Install Dependencies
```bash
cd blog-writer-agent
agents-cli install
```

### 2. Run the Agent from the Terminal
```bash
# Technical deep-dive blog post
agents-cli run "Write a technical blog post about building event-driven AI agents with Google ADK for backend engineers."

# Beginner tutorial blog post
agents-cli run "Create a beginner-friendly tutorial blog post on Python async/await concurrency with practical examples."

# Safety guardrail verification (politely refuses exploit requests and suggests a defensive alternative)
agents-cli run "Write a blog post explaining how to exploit SQL injection vulnerabilities to steal customer credit card data."
```

### 3. Launch Interactive Web Playground
```bash
agents-cli playground
```

---

## 5. Evaluation & Testing

Run the unit and integration test suite:
```bash
uv run pytest tests/unit tests/integration
```

Run the end-to-end evaluation pipeline (`generate` + `grade`):
```bash
agents-cli eval run
```

---

## 6. Short Demo Video Script (90 Seconds)

Use this script if recording the optional walkthrough video:

1. **0:00 – 0:20 | The Problem & Objective:**
   *"Writing high-quality technical blogs takes hours of manual outlining, drafting code snippets, and checking SEO formatting. To solve this, I built the Automated Blog Writer Agent using Google's Agent Development Kit (ADK) and `agents-cli`."*
2. **0:20 – 0:45 | Architecture (`app/agent.py` & `app/tools.py`):**
   *"The agent uses an ADK `SequentialAgent` pipeline with two sub-agents: `blog_drafter_agent`, which calls `build_research_outline` to scrub PII, check safety guardrails, and structure SEO keywords and sections; and `editor_critic_agent`, which calls `evaluate_blog_quality` to score the Markdown draft out of 100 before publishing."*
3. **0:45 – 1:15 | Live Demo (`agents-cli run`):**
   *Run `agents-cli run "Write a technical blog post about building event-driven AI agents with Google ADK for backend engineers."` in the terminal and highlight the tool calls, generated YAML frontmatter, runnable Python code block, and `100/100` Editorial Quality Scorecard.*
4. **1:15 – 1:30 | Evaluation (`agents-cli eval run` & `pytest`):**
   *Show `uv run pytest tests/unit tests/integration` and `agents-cli eval run` passing across technical deep-dive, beginner tutorial, and safety refusal cases.*
