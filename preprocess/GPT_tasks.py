from __future__ import annotations
import os, re, json, time, argparse
from typing import Literal, List, Dict, Any
import pandas as pd
from openai import OpenAI

from core import settings 
from core.config import GPTClassificationConfig
from core.data_utils import read_csv_safely, save_csv_safely

# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
client = OpenAI(api_key=settings.OPENAI_API_KEY)

TaskType = Literal["paragraph", "synthesis-method", "ner"]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _extract_answer_token(text: str) -> str:
    """
    Extract the <label> from phrases like:
    - "Therefore, the answer is <label>."
    - "the answer is <label>"
    Fallback: last token.
    """
    m = re.search(r"the answer is\s+([A-Za-z0-9\-\(\)\/]+)", text or "", re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    tok = (text or "").strip().split()
    return (tok[-1] if tok else "").strip().lower()


# --------------------------------------------------------------------------------------
# System prompts
# --------------------------------------------------------------------------------------
def system_prompt(task: TaskType) -> str:
    if task == "paragraph":
        return (
            "You are a catalyst expert in materials science. Classify each paragraph into one of the "
            "following four paragraph types: 'synthesis', 'system', 'performance', 'others'. "
            "Each paragraph should belong to only one category. "
            "Always end your response with the sentence: 'Therefore, the answer is <paragraph type>'."
        )

    elif task == "synthesis-method":
        return (
            "You are a catalyst synthesis expert in materials science. Classify each paragraph into one of the "
            "following seven synthesis methods: 'electrodeposition', 'sol-gel', 'solid-phase', "
            "'hydro-solvothermal', 'precipitation', 'vapor-phase', and 'others'. "
            "Each paragraph should belong to only one category. "
            "Always end your response with the sentence: 'Therefore, the answer is <synthesis method>'."
        )

    elif task == "ner":
        return (
            "You are a catalyst synthesis expert in materials science. Label each synthesis entity related to "
            "hydrothermal or solvothermal reactions in the given paragraph as one of the following five entity types: "
            "‘Solvent’, ‘Precursor’, ‘Target’, 'Additive' and ‘Substrate.\n"
            "First, identify and extract the primary ‘TARGET’ material described in a paragraph. Then, identify and extract "
            "the ‘PRECURSOR’, ‘SOLVENT’, 'Additive' and ‘SUBSTRATE’ used in the synthesis of the ‘TARGET’ material.\n"
            "The ‘PRECURSOR’ refers only to a metal precursor that provides a metal elemental source to the ‘TARGET’. "
            "Each synthesis entity should belong to only one entity type.\n"
            "Always use the answer format: \"{'entity type': 'synthesis entity', 'molecular formula', 'metal components'}\". "
            "If 'molecular formula' is ambiguous or unknown, use 'synthesis entity' as the 'molecular formula'.\n"
            "The 'molecular formula' must represent the complete chemical composition showing ALL atoms in a compound, "
            "not using abbreviated notations such as \"acac\" (acetylacetonate) or any other chemical shorthand. "
            "It is essential to display the exact atomic composition without any abbreviated functional groups or ligands.\n"
            "If the paragraph does not include synthesis entity, the entity in the response should be ‘None'."
        )
    else:
        raise ValueError(f"Unknown task type: {task}")


# --------------------------------------------------------------------------------------
# Example prompt blocks (Few-shot + CoT for classification / Few-shot only for NER)
# --------------------------------------------------------------------------------------
def example_prompt(task: TaskType) -> str:
    if task == "paragraph":
        return """### EXAMPLES (paragraph-type classification)
[Example 1]
User:
The precursor solution was prepared by dissolving nickel nitrate and cobalt nitrate in water, followed by hydrothermal treatment at 120 °C for 12 h. The product was washed and dried.
Assistant (reasoning + final):
The paragraph describes a materials preparation workflow—dissolving metal salts, hydrothermal reaction, and post-treatments (washing/drying). These steps indicate synthesis procedures, not measurement setup or performance reporting.
Therefore, the answer is synthesis.

[Example 2]
User:
Electrochemical experiments were conducted in a three-electrode system with Pt foil as the counter electrode and Ag/AgCl as the reference electrode. The electrolyte was 1.0 M KOH and all potentials were converted to RHE.
Assistant (reasoning + final):
The paragraph specifies an electrochemical measurement configuration (electrodes, electrolyte, reference) and normalization. It describes the experimental system rather than how the material was made or its performance outcomes.
Therefore, the answer is system.

[Example 3]
User:
The catalyst achieved an overpotential of 280 mV at 10 mA cm−2 with a Tafel slope of 54 mV dec−1, and showed stable performance for 20 h compared to RuO2.
Assistant (reasoning + final):
This paragraph reports quantitative activity and stability metrics with a benchmark comparison, which corresponds to performance reporting rather than synthesis or system description.
Therefore, the answer is performance.
"""

    elif task == "synthesis-method":
        return """### EXAMPLES (synthesis-method classification)
[Example 1]
User:
Metal salts were dissolved in deionized water, and the clear solution was sealed in a Teflon-lined autoclave and heated at 160 °C for 12 h. After cooling, the solid was collected and washed.
Assistant (reasoning + final):
Dissolving precursors in a solvent and reacting in a sealed Teflon-lined autoclave at elevated temperature indicates hydrothermal/solvothermal synthesis.
Therefore, the answer is hydro-solvothermal.

[Example 2]
User:
A homogeneous sol was formed by adding citric acid, followed by gelation and subsequent calcination at 500 °C to obtain the final oxide product.
Assistant (reasoning + final):
Sol formation, gelation, and calcination are the characteristic steps of sol–gel synthesis.
Therefore, the answer is sol-gel.

[Example 3]
User:
A thin layer of cobalt hydroxide nanosheets was deposited on carbon cloth through cyclic voltammetry in an aqueous electrolyte using a three-electrode cell.
Assistant (reasoning + final):
Using an electrochemical cell and applying potentials (CV) to form a film on a conductive substrate indicates electrochemical deposition.
Therefore, the answer is electrodeposition.
"""

    elif task == "ner":
        return """### EXAMPLES (NER for hydro/solvothermal synthesis)
[Example 1]
User:
Ni(NO3)2·6H2O and Fe(NO3)3·9H2O were dissolved in deionized water; the solution with a Ni foam substrate was sealed in a Teflon-lined autoclave at 160 °C for 12 h to obtain NiFe-LDH nanosheets.
Assistant:
{'entity type': 'Target', 'NiFe-LDH', 'NiFe(OH)2', 'Ni, Fe'}
{'entity type': 'Precursor', 'Ni(NO3)2·6H2O', 'Ni(NO3)2·6H2O', 'Ni'}
{'entity type': 'Precursor', 'Fe(NO3)3·9H2O', 'Fe(NO3)3·9H2O', 'Fe'}
{'entity type': 'Solvent', 'deionized water', 'H2O', ''}
{'entity type': 'Substrate', 'Ni foam', 'Ni', 'Ni'}
"""
    else:
        raise ValueError(f"Unknown task type: {task}")


# --------------------------------------------------------------------------------------
# Message builder
# --------------------------------------------------------------------------------------
def build_messages(task: TaskType, text: str, include_examples: bool) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt(task)}]

    # Put all few-shot examples in ONE block message (assistant role works well here)
    if include_examples:
        msgs.append({"role": "assistant", "content": example_prompt(task)})

    if task in ("paragraph", "synthesis-method"):
        msgs.append({
            "role": "user",
            "content": (
                "Classify the following text according to the instructions above. Provide brief reasoning and "
                "end with the exact sentence: 'Therefore, the answer is <label>'.\n\n"
                f"{text}"
            )
        })
    else:
        msgs.append({
            "role": "user",
            "content": (
                "Extract entities from the paragraph according to the system instructions. "
                "Return each entity as a separate line using the exact format:\n"
                "{'entity type': 'synthesis entity', 'molecular formula', 'metal components'}\n"
                "If multiple entities exist, output one line per entity. If none, output: None\n\n"
                f"Paragraph:\n{text}"
            )
        })
    return msgs


# --------------------------------------------------------------------------------------
# Core inference
# --------------------------------------------------------------------------------------
def process_text_gpt(messages: List[Dict[str, str]],
                     model: str,
                     temperature: float,
                     max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def process_dataset(input_file: str,
                    output_dir: str,
                    task: TaskType,
                    cfg: GPTClassificationConfig | Dict[str, Any] = None,  # kept for backward-compat; ignored
                    fewshot: bool = True,
                    preview_prompts_to: str | None = None,
                    model_override: str | None = None,
                    temperature_override: float | None = None,
                    max_tokens_override: int | None = None) -> None:
    """
    Run GPT classification/NER over a CSV.
      - input_file: CSV with 3 cols [id, text, label]; label may be dummy for inference.
      - output_dir: where predictions.csv is written.
      - task: 'paragraph' | 'synthesis-method' | 'ner'
      - cfg: deprecated (kept for backward compatibility)
      - fewshot: include example_prompt block (default=True)
      - preview_prompts_to: optional path to dump a human-readable prompt preview (for docs)
      - *_override: optional overrides for model/temperature/max_tokens
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve model params: settings → optional overrides
    model_name = model_override or settings.GPT_MODEL_NAME
    temperature = settings.GPT_TEMPERATURE if temperature_override is None else temperature_override
    max_tokens = settings.GPT_MAX_TOKENS if max_tokens_override is None else max_tokens_override

    df = read_csv_safely(input_file, header=None, usecols=[0, 1, 2])
    df.columns = ["id", "text", "label"]

    if preview_prompts_to:
        demo_msgs = build_messages(task, "<example text here>", include_examples=True)
        with open(preview_prompts_to, "w", encoding="utf-8") as f:
            f.write("# Prompt Preview (system + example_prompt + query)\n\n")
            f.write(json.dumps(demo_msgs, ensure_ascii=False, indent=2))

    results = []
    t0 = time.time()
    for _, row in df.iterrows():
        qid, text = str(row["id"]), str(row["text"])
        msgs = build_messages(task, text, include_examples=fewshot)
        completion = process_text_gpt(msgs, model_name, temperature, max_tokens)

        if task == "ner":
            pred = completion.strip()  # keep as-is (line-based)
        else:
            pred = _extract_answer_token(completion)

        results.append({"id": qid, "text": text, "prediction": pred})

        if (len(results) % 20) == 0:
            print(f"[gpt] processed {len(results)}/{len(df)}")

    out_path = os.path.join(output_dir, f"{task}_results.csv")
    save_csv_safely(pd.DataFrame(results), out_path)
    print(f"[gpt] saved: {out_path} (elapsed {time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified GPT classifier/NER (example_prompt style)")
    parser.add_argument("--task", required=True, choices=["paragraph", "synthesis-method", "ner"])
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_fewshot", action="store_true", help="disable example_prompt block")
    parser.add_argument("--preview", default=None, help="write prompt preview JSON to this path")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()

    process_dataset(
        input_file=args.input_csv,
        output_dir=args.output_dir,
        task=args.task,
        cfg=None,
        fewshot=(not args.no_fewshot),
        preview_prompts_to=args.preview,
        model_override=args.model,
        temperature_override=args.temperature,
        max_tokens_override=args.max_tokens
    )
