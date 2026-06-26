import json
from pathlib import Path
from rapidfuzz import fuzz, process

TAXONOMY_PATH = Path(__file__).parent / "skills_taxonomy.json"

with open(TAXONOMY_PATH, "r") as f:
    SKILLS_TAXONOMY = json.load(f)

def normalize_skill(raw_skill: str, threshold: int = 80) -> str | None:
    """Return the canonical skill if match exists above threshold, else None."""
    match = process.extractOne(raw_skill, SKILLS_TAXONOMY, scorer=fuzz.ratio, score_cutoff=threshold)
    return match[0] if match else None

def normalize_skills(raw_skills: list[str], threshold: int = 80) -> list[str]:
    """Normalize a list of raw skill strings, dropping unrecognized ones."""
    normalized = []
    for s in raw_skills:
        norm = normalize_skill(s, threshold)
        if norm and norm not in normalized:
            normalized.append(norm)
    return normalized