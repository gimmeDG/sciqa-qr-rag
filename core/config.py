from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from core import settings

ClassificationType = Literal["paragraph", "synthesis"]

@dataclass
class BERTClassificationConfig:
    """Task-level meta configuration for BERT classification."""
    classification_type: ClassificationType
    # meta
    label_names: list[str]
    num_labels: int
    # filenames (only "names/subdir" meta; actual paths are assembled with settings)
    input_filename: str
    output_subdir: str
    # defaults (fetched from settings; can be overridden via env in settings)
    seed: int = settings.SEED_DEFAULT
    model_name_or_path: Optional[str] = None
    batch_size: int = settings.BERT_BATCH_SIZE
    learning_rate: float = settings.BERT_LR
    epochs: int = settings.BERT_EPOCHS
    max_length: int = settings.BERT_MAX_LEN

    @staticmethod
    def create(classification_type: ClassificationType) -> "BERTClassificationConfig":
        if classification_type == "paragraph":
            labels = ["synthesis", "system", "performance", "others"]
            return BERTClassificationConfig(
                classification_type="paragraph",
                label_names=labels,
                num_labels=len(labels),
                input_filename="paragraph_raw.csv",   # flat
                output_subdir="BERT/paragraph",
            )
        elif classification_type == "synthesis":
            labels = [
                "electrodeposition",
                "sol-gel",
                "solid-phase",
                "hydro-solvothermal",
                "precipitation",
                "vapor-phase",
                "others",
            ]
            return BERTClassificationConfig(
                classification_type="synthesis",
                label_names=labels,
                num_labels=len(labels),
                input_filename="synthesis_raw.csv",   # flat
                output_subdir="BERT/synthesis",
            )
        else:
            raise ValueError(f"Unknown classification_type: {classification_type}")


@dataclass
class GPTClassificationConfig:
    """Lightweight GPT config wrapper (task selector + defaults from settings)."""
    variant: Literal["paragraph", "synthesis", "ner"]
    model: str = settings.GPT_MODEL_NAME
    temperature: float = settings.GPT_TEMPERATURE
    max_tokens: int = settings.GPT_MAX_TOKENS
    seed: int = settings.SEED_DEFAULT

    @staticmethod
    def create(variant: Literal["paragraph", "synthesis", "ner"]) -> "GPTClassificationConfig":
        return GPTClassificationConfig(variant=variant)
