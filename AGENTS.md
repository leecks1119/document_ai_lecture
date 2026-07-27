# AGENTS.md

## Project purpose

This repository contains materials and starter code for an eight-hour, beginner-friendly Document AI course.

The course teaches learners to turn one real receipt into structured JSON, validate it, review it against the source, and download `receipt_result.xlsx` from a small Streamlit prototype. Every required exercise runs in Google Colab.

## Target audience

- Beginner-level learners
- Basic Python experience
- Limited or no OCR, Document AI, Streamlit, or document automation experience

## Course scope

Each lesson must stay small enough for one 60-minute class:

- No more than three new core concepts
- One basic hands-on exercise
- One mock fallback path
- One primary learner artifact
- Two or three purposeful visuals
- No more than three formative-assessment questions
- Lessons 2–8 use no more than six minutes of new concept explanation.
- Lesson 1 may use up to 28 minutes to define the terminology and show the complete enterprise document-processing map once.
- No more than 10–15 lines of learner-written code

Do not add model training, OCR engine benchmarks, OCR ensembles, databases, authentication, production deployment, or large-scale cloud architecture to the required path.

## Development environment

- Google Colab is the only learner lab environment.
- Use the documented fixed Colab runtime for final verification.
- Keep local code compatible with Python 3.12.x.
- Streamlit 1.60.0 is the web app framework.
- PaddleOCR 3.7 with PP-OCRv5 Korean is the primary live OCR attempt in lesson 2.
- If live OCR is not working within three minutes, switch to the clearly labeled prepared result.
- PaddleOCR-VL 1.6 or one commercial VLM call is demonstrated once by the instructor with a de-identified sample.
- Do not add EasyOCR back to the 2026 course.
- Every lesson must remain completable with sample OCR text and mock JSON.
- Learners do not need API keys and do not pay for API calls.

## Source of truth

- `docs/curriculum.md`: course outcomes and lesson sequence
- `docs/rebuild_status.md`: scope, progress, and completion evidence
- `lessons/_template.md`: lesson document structure
- `lessons/_work/05_paddleocr_multimodal_research.md`: current claims, sources, and technology decisions
- `legacy_materials/`: reference only; never assume old code or claims are current

## Repository expectations

- Keep code and explanations beginner-friendly.
- Prefer plain functions, descriptive names, small steps, and short comments.
- Use one document per learner and one problem per learner.
- The core input is a de-identified lunch, coffee, or photo-gallery receipt; always provide a licensed, redacted Korean receipt fallback.
- Do not include real names, phone numbers, account numbers, API keys, or company documents.
- Make mock use visible to learners; never present mock output as a live OCR or model result.
- Preserve the original OCR text alongside cleaned and structured values.
- Use `null` when the source does not contain a requested value.
- Do not use OCR confidence alone for automatic approval.
- Use Streamlit AppTest in Colab for the required UI verification path.
- Do not require a public tunnel or local PC environment for learner exercises.
- Do not upload personal receipts to public web app links or external APIs.
- Keep the required path to one document at a time.
- The final required export is `receipt_result.xlsx`, not CSV.
- Instructor extension samples are quotation, application form, and transaction statement. Exclude purchase orders.
- Prefer native structure parsers for Excel, Word, and PowerPoint originals; prefer the text layer for text PDFs; use OCR or VLM for scans, photos, and screenshots.
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

When Streamlit behavior changes, run the app with `streamlit.testing.v1.AppTest` without launching a server.

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

The full course is complete only when all eight lessons meet these conditions and the final Streamlit mini app supports one-document upload, visible live/prepared-result paths, JSON extraction, validation, human review status, and safe `receipt_result.xlsx` export.
