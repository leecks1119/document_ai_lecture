# AGENTS.md

## Project purpose

This repository contains materials and starter code for an eight-hour, beginner-friendly Document AI course.

The course teaches learners to build a small document automation demo with OCR, structured JSON extraction, validation, CSV export, and Gradio. Google Colab is the primary hands-on environment.

## Target audience

- Beginner-level learners
- Basic Python experience
- Limited or no OCR, Document AI, Gradio, or AI coding tool experience

## Course scope

Each lesson must stay small enough for one 60-minute class:

- No more than three new core concepts
- One basic hands-on exercise
- One mock fallback path
- One primary learner artifact
- Two or three purposeful visuals
- No more than three formative-assessment questions

Do not add model training, OCR engine benchmarks, OCR ensembles, databases, authentication, production deployment, or large-scale cloud architecture to the required path.

## Development environment

- Google Colab is the primary lab environment.
- Use the documented fixed Colab runtime for final verification.
- Keep local code compatible with Python 3.12.x.
- Gradio is the primary UI.
- PaddleOCR 3.7 with PP-OCRv5 Korean is the optional live OCR path.
- PaddleOCR-VL 1.6 is the optional multimodal document parsing path.
- Do not add EasyOCR back to the 2026 course.
- Every lesson must remain completable with sample OCR text and mock JSON.
- Real generative AI API calls are optional.

## Source of truth

- `docs/curriculum.md`: course outcomes and lesson sequence
- `docs/rebuild_status.md`: scope, progress, and completion evidence
- `lessons/_template.md`: lesson document structure
- `lessons/_work/01_research_brief.md`: claims, source, and technology decisions
- `legacy_materials/`: reference only; never assume old code or claims are current

## Repository expectations

- Keep code and explanations beginner-friendly.
- Prefer plain functions, descriptive names, small steps, and short comments.
- Use one synthetic receipt example across the eight lessons.
- Do not include real names, phone numbers, account numbers, API keys, or company documents.
- Make mock use visible to learners; never present mock output as a live OCR or model result.
- Preserve the original OCR text alongside cleaned and structured values.
- Use `null` when the source does not contain a requested value.
- Do not use OCR confidence alone for automatic approval.
- Treat Gradio share links from Colab as potentially public.
- Avoid hardcoded API keys; use Colab Secrets or environment variables in optional exercises.

## Agent workflow

For course-material production, use the custom agents in this order:

1. `enterprise_docai_expert`: verify current facts, official sources, Colab constraints, and safety concerns.
2. `beginner_superstar_instructor`: turn verified material into a concise beginner lesson.
3. `senior_training_manager`: review scope, timing, clarity, reproducibility, and safety.
4. The primary Agent applies approved changes and owns final file writes.

Specialist agents should return their analysis instead of editing the same final lesson files concurrently.

## Required verification

Before claiming that course work is complete, run:

```bash
python -m compileall .
pytest
```

When notebooks or their shared code change, also run the repository notebook validator and execute every offline/mock notebook path from top to bottom.

When Gradio behavior changes, import the app successfully and verify that its demo object can be built without launching a public share link.

## Done means

A lesson is complete only when:

- Its Markdown lesson follows the lesson template.
- Its Colab notebook runs from top to bottom on the documented runtime.
- The mock path works without an OCR model download or API key.
- Expected results and troubleshooting guidance are present.
- Visuals have a clear teaching purpose and alternative text.
- Current claims link to official or primary sources.
- No secrets or real personal information are committed.
- The senior training manager reports no P0 findings and a score of at least 85.

The full course is complete only when all eight lessons meet these conditions and the final Gradio mini app supports upload, OCR or mock text, JSON extraction, validation, and CSV export.
