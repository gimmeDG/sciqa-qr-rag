from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from core import settings

ClassificationType = Literal["paragraph", "synthesis"]

@dataclass
class BERTClassificationConfig:
    """
    Task-level meta configuration for BERT classification.

    Training uses stratified train/test split (default 80:20).
    Each run uses a different random split for robustness.
    """
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
    test_size: float = 0.2  # train/test split ratio

    @staticmethod
    def create(classification_type: ClassificationType) -> "BERTClassificationConfig":
        if classification_type == "paragraph":
            labels = ["synthesis", "system", "performance", "others"]
            return BERTClassificationConfig(
                classification_type="paragraph",
                label_names=labels,
                num_labels=len(labels),
                input_filename="paragraph_testset.csv",
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
                input_filename="synthesis_testset.csv",
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
        if variant == "paragraph":
            return GPTClassificationConfig(
                variant=variant,
                model="gpt-4-1106-preview",
                temperature=0.0,
            )
        elif variant == "synthesis":
            return GPTClassificationConfig(
                variant=variant,
                model="gpt-4-1106-preview",
                temperature=0.0,
            )
        elif variant == "ner":
            return GPTClassificationConfig(
                variant=variant,
                model="gpt-4",
                temperature=0.0,
            )
        else:
            return GPTClassificationConfig(variant=variant)


@dataclass
class HoneyBeeConfig:
    """Configuration for HoneyBee (Materials Science LLM) tasks."""
    variant: Literal["paragraph", "synthesis-method", "ner"]
    base_model_path: str = str(settings.HONEYBEE_BASE_MODEL_PATH)
    lora_path: str = str(settings.HONEYBEE_LORA_PATH)
    load_in_8bit: bool = settings.HONEYBEE_LOAD_IN_8BIT
    load_in_4bit: bool = settings.HONEYBEE_LOAD_IN_4BIT
    max_new_tokens: int = settings.HONEYBEE_MAX_NEW_TOKENS
    seed: int = settings.SEED_DEFAULT

    @staticmethod
    def create(variant: Literal["paragraph", "synthesis-method", "ner"]) -> "HoneyBeeConfig":
        return HoneyBeeConfig(variant=variant)


@dataclass
class LlamaConfig:
    """Configuration for Llama 3.3 70B (via Vertex AI Model Garden MaaS)."""
    variant: Literal["paragraph", "synthesis-method", "ner"]
    project_id: str = settings.LLAMA_PROJECT_ID
    location: str = settings.LLAMA_LOCATION
    model_id: str = settings.LLAMA_MODEL_ID
    max_tokens: int = settings.LLAMA_MAX_TOKENS
    temperature: float = settings.LLAMA_TEMPERATURE
    seed: int = settings.SEED_DEFAULT

    @staticmethod
    def create(variant: Literal["paragraph", "synthesis-method", "ner"]) -> "LlamaConfig":
        return LlamaConfig(variant=variant)
