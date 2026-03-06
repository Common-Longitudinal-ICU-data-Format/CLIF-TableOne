"""LLM interpretation service for DQA results using Ollama (local)."""

import os
import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger("clif.llm")

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = """\
You are a clinical data quality expert helping researchers understand DQA \
(Data Quality Assessment) results for ICU data in the CLIF 2.1 format.

Given a summary of validation results for a CLIF table, provide a clear, \
actionable interpretation:

1. Start with a one-sentence overall assessment.
2. Explain the most critical errors first — what they mean clinically and \
what likely caused them.
3. Group related issues when possible.
4. Suggest concrete fixes or investigation steps.
5. Briefly note warnings only if they indicate a pattern.

Be concise. Use plain language accessible to clinical researchers who may not \
be database experts. Do not repeat raw numbers — the user can already see them.\
"""

SYSTEM_PROMPT_ALL = """\
You are a clinical data quality expert reviewing DQA (Data Quality Assessment) \
results across ALL tables in a CLIF 2.1 ICU dataset.

Given a summary of validation results for every table, provide a clear, \
actionable cross-table interpretation:

1. Start with a one-sentence overall dataset quality assessment.
2. Identify the most critical tables requiring immediate attention and explain why.
3. Highlight cross-table patterns (e.g., recurring column issues, shared FK problems).
4. Suggest a prioritized action plan — which tables to fix first and why.
5. Note any tables that look clean and can be considered ready.

Be concise. Use plain language accessible to clinical researchers who may not \
be database experts. Focus on patterns and priorities, not individual issue details.\
"""

SYSTEM_PROMPT_ALL_WITH_COMPARISON = """\
You are a clinical data quality expert reviewing a CLIF 2.1 ICU dataset. \
You are given two blocks of information:

1. **Comparison to Reference**: Pre-computed comparison of this dataset's \
demographics, outcomes, and treatments against the CLIF Consortium aggregate \
(~1M hospitalizations across 12 institutions). Metrics are classified as \
SIMILAR, ABOVE, or BELOW relative to the consortium.

2. **Data Quality Assessment**: DQA validation results across all tables.

Provide a clear, actionable interpretation in two sections:

## Comparison to CLIF Consortium Reference
- ALWAYS cite the exact numbers when discussing any metric (e.g., "Race Black: \
9.4% vs 19.5% consortium"). Never describe a deviation without its values.
- For ABOVE/BELOW metrics: state the values, the direction, and a brief \
plausible explanation (case mix, institution type, etc.).
- For SIMILAR metrics: briefly note alignment in 1 sentence.
- Keep this section to 4-6 sentences.

## Data Quality Assessment
- One-sentence overall quality assessment.
- Critical tables needing attention and why.
- Cross-table patterns and prioritized action plan.
- Tables that look clean and ready.

Be concise and factual. Always include specific numbers from the data provided. \
Use plain language for clinical researchers.\
"""


def is_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def curate_table_context(validation_data: dict) -> str:
    """Build a compact text summary of validation results for the LLM."""
    lines = []
    name = validation_data.get("table_name", "unknown")
    lines.append(f"Table: {name}")
    lines.append(f"Overall score: {validation_data.get('overall_pct', '?')}% "
                 f"({validation_data.get('total_passed', '?')}/{validation_data.get('total_checks', '?')} checks passed)")
    lines.append(f"Errors: {validation_data.get('error_count', 0)}, "
                 f"Warnings: {validation_data.get('warning_count', 0)}")

    # Category breakdown
    cats = validation_data.get("category_scores", {})
    if cats:
        lines.append("\nCategory breakdown:")
        for cat, scores in cats.items():
            lines.append(f"  {cat}: {scores.get('passed', '?')}/{scores.get('total', '?')}")

    # Issues list
    issues = validation_data.get("issues", [])
    if issues:
        errors = [i for i in issues if i.get("severity") == "error"]
        warnings = [i for i in issues if i.get("severity") == "warning"]

        if errors:
            lines.append(f"\nErrors ({len(errors)}):")
            for idx, e in enumerate(errors, 1):
                col = e.get("column_field", "")
                col_str = f" [{col}]" if col and col != "N/A" else ""
                desc = e.get("rule_description") or e.get("check_type", "")
                msg = e.get("message", "")
                lines.append(f"  {idx}. {e.get('category', '')}{col_str}: {desc} — {msg}")

        if warnings:
            lines.append(f"\nWarnings ({len(warnings)}):")
            for idx, w in enumerate(warnings, 1):
                col = w.get("column_field", "")
                col_str = f" [{col}]" if col and col != "N/A" else ""
                desc = w.get("rule_description") or w.get("check_type", "")
                msg = w.get("message", "")
                lines.append(f"  {idx}. {w.get('category', '')}{col_str}: {desc} — {msg}")

    return "\n".join(lines)


def curate_all_tables_context(table_validations: list[dict]) -> str:
    """Build a combined context summarising validation across all tables."""
    lines = []
    lines.append(f"CLIF 2.1 Dataset — {len(table_validations)} tables analyzed\n")

    total_errors = 0
    total_warnings = 0
    total_passed = 0
    total_checks = 0
    worst = []

    for tv in table_validations:
        name = tv.get("table_name", "unknown")
        pct = tv.get("overall_pct", 100)
        errs = tv.get("error_count", 0)
        warns = tv.get("warning_count", 0)
        tp = tv.get("total_passed", 0)
        tc = tv.get("total_checks", 0)

        total_errors += errs
        total_warnings += warns
        total_passed += tp
        total_checks += tc

        lines.append(f"--- {name} ({pct}% — {errs} errors, {warns} warnings) ---")

        issues = tv.get("issues", [])
        errors = [i for i in issues if i.get("severity") == "error"][:5]
        for idx, e in enumerate(errors, 1):
            col = e.get("column_field", "")
            col_str = f" [{col}]" if col and col != "N/A" else ""
            desc = e.get("rule_description") or e.get("check_type", "")
            msg = e.get("message", "")
            lines.append(f"  {idx}. {e.get('category', '')}{col_str}: {desc} — {msg}")
        if len([i for i in issues if i.get("severity") == "error"]) > 5:
            lines.append(f"  ... and {len([i for i in issues if i.get('severity') == 'error']) - 5} more errors")
        lines.append("")

        if errs > 0:
            worst.append((name, pct, errs))

    # Aggregate stats
    overall_pct = round(total_passed / total_checks * 100, 1) if total_checks else 100
    lines.insert(1, f"Aggregate: {overall_pct}% ({total_passed}/{total_checks} checks passed)")
    lines.insert(2, f"Total errors: {total_errors}, Total warnings: {total_warnings}")

    if worst:
        worst.sort(key=lambda x: x[1])
        lines.insert(3, f"Tables needing attention: {', '.join(f'{n} ({p}%)' for n, p, _ in worst[:5])}\n")

    return "\n".join(lines)


def curate_all_tables_context_with_comparison(
    table_validations: list[dict], comparison_text: str
) -> str:
    """Prepend consortium comparison to the standard DQA context."""
    dqa_context = curate_all_tables_context(table_validations)
    return f"{comparison_text}\n\n---\n\n{dqa_context}"


def stream_interpretation(context: str, system_prompt: str | None = None):
    """Stream an LLM interpretation via Ollama. Yields text deltas."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "system": system_prompt or SYSTEM_PROMPT,
        "prompt": context,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                yield chunk["response"]
            if chunk.get("done"):
                break
