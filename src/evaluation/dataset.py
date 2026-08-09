"""Loading and validation of the labelled evaluation set."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.config import PROJECT_ROOT

DATASET_PATH = PROJECT_ROOT / "config" / "eval_dataset.yaml"

Turn = tuple[str, str]


class InScopeCase(BaseModel):
    question: str
    expected_sources: list[str]


class ConversationCase(BaseModel):
    history: list[Turn]
    question: str


@dataclass
class EvalDataset:
    in_scope: list[InScopeCase] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    adversarial_out_of_scope: list[str] = field(default_factory=list)
    follow_ups: list[ConversationCase] = field(default_factory=list)
    off_topic_follow_ups: list[ConversationCase] = field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return (
            len(self.in_scope)
            + len(self.out_of_scope)
            + len(self.adversarial_out_of_scope)
            + len(self.follow_ups)
            + len(self.off_topic_follow_ups)
        )


def load_dataset(path: Path = DATASET_PATH) -> EvalDataset:
    """Reads and validates the evaluation set."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return EvalDataset(
        in_scope=[InScopeCase.model_validate(case) for case in raw.get("in_scope", [])],
        out_of_scope=list(raw.get("out_of_scope", [])),
        adversarial_out_of_scope=list(raw.get("adversarial_out_of_scope", [])),
        follow_ups=[
            ConversationCase.model_validate(case) for case in raw.get("follow_ups", [])
        ],
        off_topic_follow_ups=[
            ConversationCase.model_validate(case)
            for case in raw.get("off_topic_follow_ups", [])
        ],
    )
