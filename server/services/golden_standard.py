"""Compare user's table one against CLIF consortium reference data."""

import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger("clif.golden_standard")

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "data" / "overall_1113.csv"

# Metrics to compare.
# Each tuple: (display_name, section, value_type, ref_prefixes, user_prefixes)
#   value_type: "count", "pct", "median"
#   ref_prefixes / user_prefixes: list of row-label prefixes to try (first match wins)
METRICS = [
    ("Hospitalizations", "Demographics", "count",
     ["N: Hospitalizations"], ["N: Hospitalizations", "N: Encounter blocks"]),
    ("Unique patients", "Demographics", "count",
     ["N: Unique patients"], ["N: Unique patients"]),
    ("Age median", "Demographics", "median",
     ["Age at admission"], ["Age at admission"]),
    ("Sex Male", "Demographics", "pct",
     ["Sex: male"], ["Sex: male", "Sex: Male"]),
    ("Race White", "Demographics", "pct",
     ["Race: white"], ["Race: white", "Race: White"]),
    ("Race Black", "Demographics", "pct",
     ["Race: black"], ["Race: black", "Race: Black"]),
    ("Hospital mortality", "Outcomes", "pct",
     ["Hospital mortality"], ["Hospital mortality"]),
    ("Expired", "Outcomes", "pct",
     ["Expired"], ["Expired"]),
    ("Discharged to hospice", "Outcomes", "pct",
     ["Discharged to hospice"], ["Discharged to hospice"]),
    ("ICU LOS median", "Outcomes", "median",
     ["ICU length of stay"], ["ICU length of stay"]),
    ("Hospital LOS median", "Outcomes", "median",
     ["Hospital length of stay"], ["Hospital length of stay"]),
    ("CCI median", "Clinical Scores", "median",
     ["Charlson Comorbidity Index"], ["Charlson Comorbidity Index"]),
    ("Total SOFA median", "Clinical Scores", "median",
     ["Total SOFA score"], ["Total SOFA score"]),
    ("Invasive MV", "Treatments", "pct",
     ["Invasive mechanical ventilation"], ["Invasive mechanical ventilation"]),
    ("Vasopressors", "Treatments", "pct",
     ["Vasopressor encounters"], ["Vasopressor encounters"]),
    ("CRRT", "Treatments", "pct",
     ["CRRT"], ["CRRT"]),
    ("ICU encounters", "Encounter Types", "pct",
     ["ICU encounters"], ["ICU encounters"]),
    ("Advanced resp support", "Encounter Types", "pct",
     ["Advanced respiratory support"], ["Advanced respiratory support"]),
]

# Thresholds for classification
PCT_SIMILAR_THRESHOLD = 5.0      # percentage points
MEDIAN_SIMILAR_RATIO = 0.25      # 25% relative difference


def _parse_count(val: str) -> int | None:
    """Parse '1,029,400' or '89,889' into int."""
    try:
        return int(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_pct(val: str) -> float | None:
    """Extract percentage from '208,813 (25.8%)' or '25.8%'."""
    m = re.search(r"([\d.]+)%", val)
    return float(m.group(1)) if m else None


def _parse_median(val: str) -> float | None:
    """Extract median from '66 [47,79]' or '1.7 [0.8,5]'."""
    m = re.match(r"^\s*([\d.]+)", val.strip())
    return float(m.group(1)) if m else None


def _classify(user_val: float, ref_val: float, value_type: str) -> str:
    """Classify difference as SIMILAR, ABOVE, or BELOW."""
    if value_type == "pct":
        diff = user_val - ref_val
        if abs(diff) <= PCT_SIMILAR_THRESHOLD:
            return "SIMILAR"
        return "ABOVE" if diff > 0 else "BELOW"
    else:
        # median or count — use relative difference
        if ref_val == 0:
            return "ABOVE" if user_val > 0 else "SIMILAR"
        ratio = abs(user_val - ref_val) / ref_val
        if ratio <= MEDIAN_SIMILAR_RATIO:
            return "SIMILAR"
        return "ABOVE" if user_val > ref_val else "BELOW"


def _load_csv_column(path: Path, col_index: int) -> dict[str, str]:
    """Load a CSV and return {row_label: value} for the given column index."""
    result = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) <= col_index:
                    continue
                label = row[0].strip()
                value = row[col_index].strip() if row[col_index] else ""
                if label and value:
                    result[label] = value
    except Exception as e:
        logger.warning("Failed to parse CSV %s: %s", path, e)
    return result


def _load_reference() -> dict[str, str]:
    """Load consortium aggregate column (index 1) from reference CSV."""
    if not REFERENCE_PATH.exists():
        logger.warning("Reference data not found: %s", REFERENCE_PATH)
        return {}
    return _load_csv_column(REFERENCE_PATH, 1)


def _load_user_tableone(csv_path: Path) -> dict[str, str]:
    """Load user's table one — column index 1 (the 'Overall' column)."""
    if not csv_path.exists():
        return {}
    return _load_csv_column(csv_path, 1)


def _match_metric(row_labels: dict[str, str], prefixes: list[str]) -> str | None:
    """Find a row label matching any of the given prefixes (case-insensitive)."""
    for prefix in prefixes:
        prefix_lower = prefix.lower()
        for label, value in row_labels.items():
            if label.lower().lstrip().startswith(prefix_lower):
                return value
    return None


def build_comparison_context(user_csv_path: Path) -> str | None:
    """Build a compact comparison text between user data and consortium reference.

    Returns None if comparison cannot be performed (no reference or no user data).
    """
    ref_data = _load_reference()
    if not ref_data:
        return None

    user_data = _load_user_tableone(user_csv_path)
    if not user_data:
        return None

    sections: dict[str, list[str]] = {}
    matched = 0

    for display_name, section, value_type, ref_prefixes, user_prefixes in METRICS:
        ref_val_str = _match_metric(ref_data, ref_prefixes)
        user_val_str = _match_metric(user_data, user_prefixes)

        if not ref_val_str:
            continue

        if not user_val_str:
            sections.setdefault(section, []).append(
                f"  {display_name}: N/A (yours) vs {ref_val_str} (consortium)"
            )
            continue

        # Parse based on type
        if value_type == "count":
            user_num = _parse_count(user_val_str)
            ref_num = _parse_count(ref_val_str)
            if user_num is not None and ref_num is not None:
                sections.setdefault(section, []).append(
                    f"  {display_name}: {user_val_str} (yours) vs {ref_val_str} (consortium)"
                )
                matched += 1
        elif value_type == "pct":
            user_pct = _parse_pct(user_val_str)
            ref_pct = _parse_pct(ref_val_str)
            if user_pct is not None and ref_pct is not None:
                label = _classify(user_pct, ref_pct, "pct")
                sections.setdefault(section, []).append(
                    f"  {display_name}: {user_pct}% vs {ref_pct}% -- {label}"
                )
                matched += 1
        elif value_type == "median":
            user_med = _parse_median(user_val_str)
            ref_med = _parse_median(ref_val_str)
            if user_med is not None and ref_med is not None:
                label = _classify(user_med, ref_med, "median")
                sections.setdefault(section, []).append(
                    f"  {display_name}: {user_med} vs {ref_med} -- {label}"
                )
                matched += 1

    if matched < 3:
        logger.info("Too few metrics matched (%d), skipping comparison", matched)
        return None

    lines = [
        "COMPARISON TO CLIF CONSORTIUM (~1M hospitalizations, 12 institutions)",
        "",
    ]
    for section_name in [
        "Demographics", "Outcomes", "Encounter Types",
        "Clinical Scores", "Treatments",
    ]:
        items = sections.get(section_name)
        if items:
            lines.append(f"{section_name}:")
            lines.extend(items)
            lines.append("")

    lines.append(
        "Deviations from consortium are not necessarily errors "
        "-- single-site data naturally varies by patient population, "
        "case mix, and institutional practices."
    )
    return "\n".join(lines)
