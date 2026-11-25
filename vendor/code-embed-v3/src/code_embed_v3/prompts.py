"""Task families and their query-side instruction prompts.

Documents are always encoded without a prompt so one corpus index serves every task family.
"""

from __future__ import annotations

from typing import Final, Literal

Family = Literal["nl2code", "code2nl", "code2code", "techqa", "text2sql", "code_edit", "multiturn"]

QUERY_PROMPTS: Final[dict[Family, str]] = {
    "nl2code": "Find the code that answers this query: ",
    "code2nl": "Find the description of this code: ",
    "code2code": "Find code equivalent to this code: ",
    "techqa": "Find the answer to this programming question: ",
    "text2sql": "Find the SQL query for this request: ",
    "code_edit": "Find the code change described by: ",
    "multiturn": "Find the reply that continues this conversation: ",
}

# Every task in MTEB(Code, v1), CoIR and CoREB(v1), mapped from its mteb metadata description.
MTEB_TASK_FAMILY: Final[dict[str, Family]] = {
    "AppsRetrieval": "nl2code",
    "CodeEditSearchRetrieval": "code_edit",
    "CodeFeedbackMT": "multiturn",
    "CodeFeedbackST": "techqa",
    "CodeSearchNetCCRetrieval": "code2code",
    "CodeSearchNetRetrieval": "nl2code",
    "CodeTransOceanContest": "code2code",
    "CodeTransOceanDL": "code2code",
    "CosQA": "nl2code",
    "COIRCodeSearchNetRetrieval": "code2nl",
    "StackOverflowQA": "techqa",
    "SyntheticText2SQL": "text2sql",
    "CorebC2TRetrieval": "code2nl",
    "CorebC2CRetrieval": "code2code",
    "CorebT2CRetrieval": "nl2code",
    "CorebC2TReranking": "code2nl",
    "CorebC2CReranking": "code2code",
    "CorebT2CReranking": "nl2code",
}


def model_prompts() -> dict[str, str]:
    """Prompts stored on the SentenceTransformer, addressable via ``encode(..., prompt_name=family)``."""
    return {family: prompt for family, prompt in QUERY_PROMPTS.items()}


def mteb_model_prompts() -> dict[str, str]:
    """``{task}-query`` keys, the highest-priority form in mteb's prompt resolution."""
    return {f"{task}-query": QUERY_PROMPTS[family] for task, family in MTEB_TASK_FAMILY.items()}


def training_prompts(
    dataset_families: dict[str, Family], anchor_column: str = "anchor"
) -> dict[str, dict[str, str]]:
    """Per-dataset prompt mapping for ``SentenceTransformerTrainingArguments(prompts=...)``."""
    return {name: {anchor_column: QUERY_PROMPTS[family]} for name, family in dataset_families.items()}
