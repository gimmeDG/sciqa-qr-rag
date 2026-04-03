"""
HoneyBee Tasks
==============
Materials Science LLM (HoneyBee) for:
- Paragraph Classification
- Synthesis Method Classification
- Named Entity Recognition (NER)

Uses 8-bit quantization for 12GB VRAM compatibility.
"""
from __future__ import annotations
import os
import re
import json
import time
import argparse
from typing import Literal, List, Dict, Any, Optional
from collections import defaultdict

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import confusion_matrix, classification_report

from core import settings
from core.config import HoneyBeeConfig
from core.data_utils import read_csv_safely, save_csv_safely, set_seed

TaskType = Literal["paragraph", "synthesis-method", "ner"]

# Global model cache (singleton pattern for efficiency)
_MODEL_CACHE: Dict[str, Any] = {}


# --------------------------------------------------------------------------------------
# NER Evaluation Helpers
# --------------------------------------------------------------------------------------
def _clean_json_string(json_str: str) -> str:
    json_str = json_str.replace("\\xa0", " ").replace("\\", "\\\\")
    json_str = json_str.replace("'", '"')
    json_str = json_str.replace("\n", " ").replace("\r", " ")
    json_str = re.sub(r'\s+', ' ', json_str)
    return json_str


def _extract_ner_entities(content: str) -> List[Dict[str, Any]]:
    """Extract entities from NER output."""
    entity_pattern = r"\{'(?P<type>[^']+)': '(?P<entity>[^']+)', '(?P<formula>[^']+)', '(?P<metal>[^']+)'\}"
    matches = re.finditer(entity_pattern, content)
    extracted = []
    for match in matches:
        metals = [m.strip() for m in match.group("metal").split(",") if m.strip()] if match.group("metal") else []
        extracted.append({
            "type": match.group("type"),
            "entity": match.group("entity"),
            "formula": match.group("formula"),
            "metals": metals,
        })
    return extracted


def _filter_ner_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
    """Filter entities based on TARGET/PRECURSOR rules."""
    targets = [e for e in entities if e['type'].upper() == 'TARGET']
    if len(targets) != 1 or any("MOF" in t['entity'].upper() for t in targets):
        return None

    target_metals = set(targets[0]['metals'])
    filtered = []
    for entity in entities:
        if entity['type'].upper() == 'TARGET' and not entity['metals']:
            continue
        if entity['type'].upper() == 'PRECURSOR':
            if not entity['metals'] or not set(entity['metals']).intersection(target_metals):
                continue
        filtered.append(entity)
    return filtered


def _parse_filtered_to_labels(entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert filtered entities to label format."""
    valid_keys = ['target', 'precursor', 'substrate', 'solvent', 'additive']
    return [{e['type'].lower(): e['entity']} for e in entities if e['type'].lower() in valid_keys]


def _extract_actual_ner_labels(label_str: str) -> List[Dict[str, str]]:
    """Extract actual labels from CSV label column."""
    cleaned = _clean_json_string(str(label_str).strip())
    if not cleaned.startswith("[") and not cleaned.endswith("]"):
        cleaned = "[" + cleaned + "]"
    try:
        return json.loads(cleaned)
    except:
        return []


def _evaluate_ner_performance(actual_list: List, predicted_list: List, match_type: str = 'r') -> Dict[str, Any]:
    """Evaluate NER performance with relaxed or exact matching."""
    keys = ['target', 'precursor', 'substrate', 'solvent', 'additive']
    performance = {k: {'TP': 0, 'FP': 0, 'FN': 0, 'Total': 0} for k in keys}

    for actual, predicted in zip(actual_list, predicted_list):
        if actual is None or predicted is None:
            continue
        actual_dict = defaultdict(set)
        predicted_dict = defaultdict(set)

        for item in actual:
            for key, value in item.items():
                nk = key.lower()
                if nk in keys:
                    actual_dict[nk].add(value.lower())
                    performance[nk]['Total'] += 1

        for item in predicted:
            for key, value in item.items():
                nk = key.lower()
                if nk in keys:
                    predicted_dict[nk].add(value.lower())

        for key in keys:
            matched_actual, matched_predicted = set(), set()
            tp = 0
            if match_type == 'e':
                tp = len(actual_dict[key] & predicted_dict[key])
                matched_actual = matched_predicted = actual_dict[key] & predicted_dict[key]
            else:
                for act in actual_dict[key]:
                    for pred in predicted_dict[key]:
                        if pred in act or act in pred:
                            tp += 1
                            matched_actual.add(act)
                            matched_predicted.add(pred)
            performance[key]['TP'] += tp
            performance[key]['FP'] += len(predicted_dict[key] - matched_predicted)
            performance[key]['FN'] += len(actual_dict[key] - matched_actual)

    total_f1, weighted_f1, correct, total_pred = 0, 0, 0, 0
    total_labels = sum(performance[k]['Total'] for k in keys)

    print("\n" + "="*60)
    print("NER Evaluation Results (Relaxed Match)" if match_type == 'r' else "NER Evaluation Results (Exact Match)")
    print("="*60)

    for key in keys:
        tp, fp, fn = performance[key]['TP'], performance[key]['FP'], performance[key]['FN']
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        correct += tp
        total_pred += tp + fp + fn
        total_f1 += f1 / 5
        if total_labels > 0:
            weighted_f1 += f1 * performance[key]['Total'] / total_labels
        print(f"{key}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Total={performance[key]['Total']}")

    accuracy = correct / total_pred if total_pred > 0 else 0
    print("-"*60)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1-Score: {total_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")
    print("="*60)

    return {'accuracy': accuracy, 'macro_f1': total_f1, 'weighted_f1': weighted_f1, 'per_class': performance}


# --------------------------------------------------------------------------------------
# Model Loading (8-bit / 4-bit quantization)
# --------------------------------------------------------------------------------------
def load_honeybee_model(cfg: HoneyBeeConfig):
    """Load HoneyBee model with LoRA adapter and quantization."""
    cache_key = f"{cfg.base_model_path}_{cfg.lora_path}_{cfg.load_in_8bit}_{cfg.load_in_4bit}"

    if cache_key in _MODEL_CACHE:
        print("[honeybee] using cached model")
        return _MODEL_CACHE[cache_key]

    print(f"[honeybee] loading base model: {cfg.base_model_path}")
    print(f"[honeybee] loading LoRA adapter: {cfg.lora_path}")

    # Quantization config
    if cfg.load_in_4bit:
        print("[honeybee] using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.load_in_8bit:
        print("[honeybee] using 8-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        print("[honeybee] using FP16 (no quantization)")
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_path,
        trust_remote_code=True,
        use_fast=False,  # Use slow tokenizer to avoid SentencePiece conversion issues
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with quantization
    device_map = "auto" if settings.USE_GPU else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16 if bnb_config is None else None,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print("[honeybee] applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, cfg.lora_path)
    model.eval()

    print(f"[honeybee] model loaded successfully (device: {next(model.parameters()).device})")

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


# --------------------------------------------------------------------------------------
# System prompts (same as GPT_tasks.py)
# --------------------------------------------------------------------------------------
def system_prompt(task: TaskType) -> str:
    if task == "paragraph":
        return (
            "As a catalyst expert in materials science, classify each paragraph into one of the following "
            "four paragraph types: 'synthesis', 'system', 'performance' and 'others'. "
            "Each paragraph should belong to only one category. "
            "Always include a sentence in the format of 'the answer is paragraph type' at the end of the answer."
        )
    elif task == "synthesis-method":
        return (
            "As a catalyst synthesis expert in materials science, classify each paragraph into one of the following "
            "seven synthesis methods: 'electrodeposition', 'sol-gel', 'solid-phase', 'hydro-solvothermal', "
            "'precipitation', 'vapor-phase' and 'others'. "
            "Each paragraph should belong to only one category. "
            "Always include a sentence in the format of 'the answer is synthesis method' at the end of the answer."
        )
    elif task == "ner":
        return (
            "As a catalyst synthesis expert in materials science, label each synthesis entity related to "
            "hydrothermal or solvothermal reactions in the given paragraph as one of the following five entity types: "
            "'Solvent', 'Precursor', 'Target', 'Additive' and 'Substrate. \n"
            "First, identify and extract the primary 'TARGET' material described in a paragraph. Then, identify and extract "
            "the 'PRECURSOR', 'SOLVENT', 'Additive' and 'SUBSTRATE' used in the synthesis of the 'TARGET' material. \n"
            "The 'PRECURSOR' refers only to a metal precursor that provides a metal elemental source to the 'TARGET'. "
            "Each synthesis entity should belong to only one entity type. \n"
            "Always use the answer format: \"{'entity type': 'synthesis entity', 'molecular formula', 'metal components'}\". "
            "If 'molecular formula' is ambiguous or unknown, use 'synthesis entity' as the 'molecular formula'. \n"
            "The 'molecular formula' must represent the complete chemical composition showing ALL atoms in a compound, "
            "not using abbreviated notations such as \"acac\" (acetylacetonate) or any other chemical shorthand. "
            "It is essential to display the exact atomic composition without any abbreviated functional groups or ligands.\n"
            "If the paragraph does not include synthesis entity, the entity in the response should be 'None'."
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# --------------------------------------------------------------------------------------
# Few-shot examples (CoT format matching GPT/Llama, 1 per category to fit 2048 limit)
# --------------------------------------------------------------------------------------
def get_examples(task: TaskType) -> List[Dict[str, str]]:
    """Get few-shot examples with CoT reasoning (matching GPT/Llama format, 1 per category)."""
    if task == "paragraph":
        # 4 classes x 1 example each = 4 pairs with CoT (~500 tokens), leaves ~1500 for input
        return [
            # === SYNTHESIS (1) ===
            {"role": "user", "content": "PD-Co(OH)2/CC was prepared as follows. The carbon cloth was cleaned by sonication and immersed in 0.1 M Co(NO3)2 solution for electrodeposition. Electrodeposition was carried out by potentiodynamic deposition in a three-electrode system. The coated carbon cloth was rinsed with water and dried in a vacuum oven."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the preparation procedure for PD-Co(OH)2/CC, including cleaning, electrodeposition process, and drying. This is a typical description of how a material is synthesized. Therefore, the answer is synthesis."},
            # === SYSTEM (1) ===
            {"role": "user", "content": "Electrochemical tests were carried out on a Princeton Applied Research Parstat 3000-DX with a three-electrode system in N2-saturated 1.0 M KOH. The Ni(OH)2@1T-MoS2 NWAs was directly served as working electrode. A graphite rod and a Hg/HgO electrode were served as counter and reference electrodes."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the experimental setup and procedures for electrochemical tests, including equipment, electrode configuration, and electrolyte conditions. It focuses on the experimental system and methodology. Therefore, the answer is system."},
            # === OTHERS (1) ===
            {"role": "user", "content": "DFT calculations were performed to understand the electronic structure and catalytic mechanism. The adsorption energies of OH*, O*, and OOH* intermediates were computed on the (110) surface of the catalyst."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes theoretical DFT calculations to understand electronic structure and mechanism, focusing on computational analysis rather than synthesis, system setup, or performance results. Therefore, the answer is others."},
            # === PERFORMANCE (1) ===
            {"role": "user", "content": "The V-Ni3N/NF exhibits high catalytic OER activity, achieving a current density of 10 mA cm-2 at a low potential of 1.519 V, which is smaller than the values for Ni3N/NF (1.547 V) and RuO2 (1.504 V). It delivers high current densities of 200 and 400 mA cm-2 at low potentials."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the OER catalytic activity with specific metrics like current density and potential values, comparing performance with other materials. The focus is on electrical performance evaluation. Therefore, the answer is performance."},
        ]
    elif task == "synthesis-method":
        # 7 classes x 1 example each = 7 pairs with CoT (~700 tokens), leaves ~1300 for input
        return [
            # === ELECTRODEPOSITION (1) ===
            {"role": "user", "content": "Ni foam was sonicated in acetone, HCl, water, ethanol to remove impurities. The electrodepositing of NiFe composites onto NF was undertaken in a three-electrode system containing nickel and iron nitrates at room temperature, using NF as the working electrode, Pt as the counter electrode, Ag/AgCl as reference electrode. The deposition potential was −1.0 V vs. Ag/AgCl for 300 s."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process involves cleaning the Ni foam, followed by using a three-electrode system to deposit NiFe composites onto the Ni foam using a specific voltage. This indicates the use of an electrical current to induce the deposition of material onto a substrate. Therefore, the answer is electrodeposition."},
            # === SOL-GEL (1) ===
            {"role": "user", "content": "Ni and Cr precursors with Fe precursors as dopants are dissolved in N,N-dimethylformamide under 70°C for 24h to form a viscous gel state, then the prepared gel compounds were annealed under 700°C for 3h. The compound was left to cool to room temperature, forming a brown color powder."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process involves dissolving metal precursors in a solvent to form a gel, followed by annealing to obtain the final product. This method is characteristic of the sol-gel process, which involves transitioning from a liquid 'sol' to a solid 'gel' phase. Therefore, the answer is sol-gel."},
            # === SOLID-PHASE (1) ===
            {"role": "user", "content": "Powder precursors SrCO3, Ir metal, and ZnO were mixed, ground, and preheated in an alumina crucible at 800°C for 12h in air. The resulting powders were then ground, pelletized, and calcined in air at 1000°C for 48h, before a final grinding, pelletizing, and calcining at 1100°C for 24h."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The method involves mixing, grinding, and heating solid powder precursors, followed by a series of calcination steps. This process does not involve a solvent or liquid phase, nor vapor or electrochemical deposition. Given the use of solid precursors and thermal treatment, this is a solid-phase synthesis. Therefore, the answer is solid-phase."},
            # === HYDRO-SOLVOTHERMAL (1) ===
            {"role": "user", "content": "Ni(NO3)2·6H2O, Fe(NO3)3·9H2O, urea, and SDS were dissolved in 40 mL DI water under magnetic stirring. The solution was transferred to a 50 mL Teflon-lined stainless-steel autoclave containing CNTs/CP and heated at 120°C in an oven for 8h. After cooling, the sample was washed with water and ethanol."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process involves dissolving precursors in water, transferring the solution to a Teflon-lined autoclave, and heating the mixture under pressure. This method is characteristic of hydrothermal synthesis using an autoclave at elevated temperatures. Therefore, the answer is hydro-solvothermal."},
            # === PRECIPITATION (1) ===
            {"role": "user", "content": "Ni(NO3)2·6H2O and FeCl2·4H2O were dissolved in distilled water with vigorous stirring. Na4P2O7·10H2O was dissolved in distilled water. Then, the two solutions were mixed with vigorous stirring and kept at room temperature overnight. The bright-green colored precipitates were collected through centrifuge and washed with water."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The synthesis process involves dissolving precursors in water, mixing them to form a precipitate, and then collecting and drying the precipitate. This method is characterized by the formation of solid particles from a solution, which is the hallmark of co-precipitation technique. Therefore, the answer is precipitation."},
            # === VAPOR-PHASE (1) ===
            {"role": "user", "content": "CoFe precursor was placed at the downstream side of a porcelain boat, and ammonium fluoride (NH4F) was at the upstream side. The porcelain boat was put in a tube furnace. The annealing temperature was set at 320°C for 2h at N2 atmosphere with a heating rate of 3°C min−1 and N2 flow of 10 cc min−1."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The synthesis involves placing a precursor in a tube furnace, heating it under nitrogen atmosphere, and using ammonium fluoride for fluorination. This process uses a tube furnace and gas flow, which aligns with vapor-phase synthesis where reactions occur in the gas phase at elevated temperatures. Therefore, the answer is vapor-phase."},
            # === OTHERS (1) ===
            {"role": "user", "content": "The IrO2-Ta2O5 layer was coated on Ti substrates by drop-casting a 2-propanol solution. The precursor solution was dropped on Ti substrates and dried in an oven at 80°C. After repeating three times, the thermal decomposition was performed by calcination in air at 450°C for 1h."},
            {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process involves dissolving metal precursors in a solvent, drop-casting the solution onto substrates, and then performing thermal treatment to decompose the precursors. This method does not fit neatly into electrodeposition, sol-gel, solid-phase, hydro-solvothermal, precipitation, or vapor-phase synthesis. Therefore, the answer is others."},
        ]
    elif task == "ner":
        # NER: 5-shot examples (reduced from 10 to fit HoneyBee's 2048 token limit)
        # Each example covers different entity combinations
        return [
            # Example 1: PRECURSOR, SOLVENT, TARGET, ADDITIVE
            {"role": "user", "content": "In a typical experiment, RuCl3·xH2O (13.0 mg) and Na2IrCl6·6H2O (40.5 mg) were added to a water/methanol mixed dispersion of ZIF-8 nanocubes. After solvothermal treatment (80 °C for 2 h), the RuIrZnOx-U were obtained."},
            {"role": "assistant", "content": "{'PRECURSOR': 'RuCl3·xH2O', 'RuCl3·xH2O', 'Ru'}, {'PRECURSOR': 'Na2IrCl6·6H2O', 'Na2IrCl6·6H2O', 'Na, Ir'}, {'SOLVENT': 'water', 'H2O', 'None'}, {'SOLVENT': 'methanol', 'CH3OH', 'None'}, {'TARGET': 'RuIrZnOx', 'RuIrZnOx', 'Ru, Ir, Zn'}, {'ADDITIVE': 'ZIF-8 nanocubes', 'C8H12N4Zn', 'Zn'}"},
            # Example 2: TARGET, PRECURSOR, SOLVENT, ADDITIVE (different chemicals)
            {"role": "user", "content": "The Ni6MnO8 microflower was prepared by a simple solvothermal reaction. DMF (30 mL) and acetone (30 mL) were mixed. Ni(NO3)2·6H2O and Mn(NO3)2·4H2O were dissolved and stirred. After the addition of graphene oxide solution, the mixture was added to an autoclave and held at 160 °C for 4 h."},
            {"role": "assistant", "content": "{'TARGET': 'Ni6MnO8', 'Ni6MnO8', 'Ni, Mn'}, {'SOLVENT': 'DMF', 'C3H7NO', 'None'}, {'SOLVENT': 'acetone', 'C3H6O', 'None'}, {'PRECURSOR': 'Ni(NO3)2·6H2O', 'Ni(NO3)2·6H2O', 'Ni'}, {'PRECURSOR': 'Mn(NO3)2·4H2O', 'Mn(NO3)2·4H2O', 'Mn'}, {'ADDITIVE': 'graphene oxide', '(C2OH)n', 'None'}"},
            # Example 3: TARGET, PRECURSOR, ADDITIVE, SOLVENT, SUBSTRATE (with substrate)
            {"role": "user", "content": "NiFe–MOF was synthesized through solvothermal method. 580 mg of nickel nitrate hexahydrate, 270 mg of ferric chloride hexahydrate, and 166 mg of 1,4-benzenedicarboxylate were dispersed in 36 mL ethanol. A piece of precleaned nickel foam was immersed and maintained at 150 °C for 6 h."},
            {"role": "assistant", "content": "{'TARGET': 'NiFe–MOF', 'NiFe–MOF', 'Ni, Fe'}, {'PRECURSOR': 'nickel nitrate hexahydrate', 'Ni(NO3)2·6H2O', 'Ni'}, {'PRECURSOR': 'ferric chloride hexahydrate', 'FeCl3·6H2O', 'Fe'}, {'ADDITIVE': '1,4-benzenedicarboxylate', 'C8H4O4', 'None'}, {'SOLVENT': 'ethanol', 'CH3CH2OH', 'None'}, {'SUBSTRATE': 'nickel foam', 'Ni', 'Ni'}"},
            # Example 4: SUBSTRATE, TARGET, SOLVENT, PRECURSOR, ADDITIVE (Ni foam substrate)
            {"role": "user", "content": "Ni foam is sonicated in HCl solution to remove the oxide layer. The CoMoO4/NF precursor is prepared by hydrothermal process. 50 mL of distilled water is used to dissolve Co(NO3)2·6H2O and (NH4)6Mo7O24·4H2O. 1.2 g CO(NH2)2 is added as surfactant. The solution is transferred to an autoclave with treated NF."},
            {"role": "assistant", "content": "{'SUBSTRATE': 'Ni foam', 'Ni', 'Ni'}, {'TARGET': 'CoMoO4/NF', 'CoMoO4/NF', 'Co, Mo, Ni'}, {'SOLVENT': 'distilled water', 'H2O', 'None'}, {'PRECURSOR': 'Co(NO3)2·6H2O', 'Co(NO3)2·6H2O', 'Co'}, {'PRECURSOR': '(NH4)6Mo7O24·4H2O', '(NH4)6Mo7O24·4H2O', 'Mo'}, {'ADDITIVE': 'CO(NH2)2', 'CO(NH2)2', 'None'}"},
            # Example 5: PRECURSOR, ADDITIVE, TARGET, SUBSTRATE, SOLVENT (with NGF substrate)
            {"role": "user", "content": "Typically, 0.3 mmol (NH4)6Mo7O24·4H2O, 12 mmol CH4N2S, and 23 mg NGF were mixed into 70 mL distilled water and stirred for 30 min, then transferred into an autoclave and heated at 180 °C for 12 h to obtain MoS2/NGF precipitate."},
            {"role": "assistant", "content": "{'PRECURSOR': '(NH4)6Mo7O24·4H2O', '(NH4)6Mo7O24·4H2O', 'Mo'}, {'ADDITIVE': 'CH4N2S', 'CH4N2S', 'None'}, {'TARGET': 'MoS2/NGF', 'MoS2/NGF', 'Mo'}, {'SUBSTRATE': 'NGF', 'CxNy', 'None'}, {'SOLVENT': 'distilled water', 'H2O', 'None'}"},
        ]
    else:
        raise ValueError(f"Unknown task: {task}")


# --------------------------------------------------------------------------------------
# Prompt Templates (matching GPT format with few-shot examples)
# --------------------------------------------------------------------------------------
def build_instruction_prompt(task: TaskType, text: str, include_examples: bool = True) -> str:
    """Build prompt with system instruction and few-shot examples (matching GPT format)."""
    instruction = system_prompt(task)

    # Build few-shot examples string
    examples_str = ""
    if include_examples:
        examples = get_examples(task)
        for ex in examples:
            if ex["role"] == "user":
                examples_str += f"\n### Input:\n{ex['content']}\n"
            elif ex["role"] == "assistant":
                examples_str += f"### Response:\n{ex['content']}\n"

    # HoneyBee instruction format with few-shot
    prompt = f"""### Instruction:
{instruction}
{examples_str}
### Input:
{text}

### Response:
"""
    return prompt


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    return_token_counts: bool = False,
):
    """Generate response from HoneyBee model.

    Args:
        return_token_counts: If True, returns (response, token_info) tuple

    Returns:
        str if return_token_counts=False
        tuple[str, Dict[str, int]] if return_token_counts=True
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_token_count = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    output_token_count = outputs[0].shape[0] - input_token_count
    response = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)

    if return_token_counts:
        token_info = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "total_tokens": input_token_count + output_token_count,
        }
        return response.strip(), token_info

    return response.strip()


def extract_answer_token(text: str, task: TaskType = "paragraph") -> str:
    """Extract classification label from response with validation against valid labels."""
    # Define valid labels per task
    VALID_LABELS = {
        "paragraph": {"synthesis", "system", "performance", "others"},
        "synthesis-method": {"electrodeposition", "sol-gel", "solid-phase", "hydro-solvothermal",
                            "precipitation", "vapor-phase", "others"},
        "ner": set(),  # NER uses different parsing
    }
    valid_set = VALID_LABELS.get(task, set())

    # Pattern: "Therefore, the answer is <label>."
    m = re.search(r"the answer is\s+([A-Za-z0-9\-]+)", text or "", re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().lower()
        if valid_set and candidate in valid_set:
            return candidate
        # If candidate is not valid, try to find valid label anywhere in text

    # Fallback: search for any valid label in the text (priority order)
    text_lower = (text or "").lower()
    if valid_set:
        # Check each valid label (prefer more specific matches)
        for label in valid_set:
            if label in text_lower:
                return label

    # Last resort: return last word (may be invalid, will be flagged in evaluation)
    tokens = (text or "").strip().split()
    return (tokens[-1] if tokens else "").strip().lower().rstrip(".")


# --------------------------------------------------------------------------------------
# Dataset Processing
# --------------------------------------------------------------------------------------
def process_dataset(
    input_file: str,
    output_dir: str,
    task: TaskType,
    cfg: Optional[HoneyBeeConfig] = None,
    preview_prompts_to: Optional[str] = None,
) -> None:
    """
    Run HoneyBee inference over a CSV dataset.

    Args:
        input_file: CSV with columns [id, text, label] (label may be dummy)
        output_dir: Output directory for predictions
        task: 'paragraph' | 'synthesis-method' | 'ner'
        cfg: HoneyBeeConfig (optional, uses defaults if None)
        preview_prompts_to: Optional path to save prompt preview
    """
    if cfg is None:
        cfg = HoneyBeeConfig.create(task)

    set_seed(cfg.seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_honeybee_model(cfg)

    # Load data (columns: id, label, text)
    df = read_csv_safely(input_file, header=None, usecols=[0, 1, 2])
    df.columns = ["id", "label", "text"]

    # Preview prompt (optional)
    if preview_prompts_to:
        demo_prompt = build_instruction_prompt(task, "<example text here>")
        with open(preview_prompts_to, "w", encoding="utf-8") as f:
            f.write("# HoneyBee Prompt Preview\n\n")
            f.write(demo_prompt)

    results = []
    actual_labels_list = []
    predicted_labels_list = []
    token_stats = {"input": [], "output": [], "total": [], "max_input": 0, "max_input_idx": -1}
    t0 = time.time()

    for idx, row in df.iterrows():
        qid, text, label = str(row["id"]), str(row["text"]), str(row["label"])

        # HoneyBee uses reduced few-shot examples (2 per class) to fit 2048 token limit
        prompt = build_instruction_prompt(task, text, include_examples=True)
        response, token_info = generate_response(model, tokenizer, prompt, cfg.max_new_tokens, return_token_counts=True)

        # Track token statistics
        token_stats["input"].append(token_info["input_tokens"])
        token_stats["output"].append(token_info["output_tokens"])
        token_stats["total"].append(token_info["total_tokens"])
        if token_info["input_tokens"] > token_stats["max_input"]:
            token_stats["max_input"] = token_info["input_tokens"]
            token_stats["max_input_idx"] = len(results)

        if task == "ner":
            pred = response  # Keep raw output for NER
            # Extract and filter entities for evaluation
            entities = _extract_ner_entities(response)
            filtered = _filter_ner_entities(entities)
            pred_labels = _parse_filtered_to_labels(filtered) if filtered else []
            actual_labels = _extract_actual_ner_labels(label)
            actual_labels_list.append(actual_labels)
            predicted_labels_list.append(pred_labels)
        else:
            pred = extract_answer_token(response, task)
            actual_labels_list.append(label.strip().lower())
            predicted_labels_list.append(pred.strip().lower() if pred else "")

        results.append({
            "id": qid,
            "text": text,
            "label": label,
            "prediction": pred,
            "input_tokens": token_info["input_tokens"],
            "output_tokens": token_info["output_tokens"],
        })

        if (len(results) % 10) == 0:
            elapsed = time.time() - t0
            avg_time = elapsed / len(results)
            remaining = avg_time * (len(df) - len(results))
            avg_in = sum(token_stats["input"]) / len(token_stats["input"])
            avg_out = sum(token_stats["output"]) / len(token_stats["output"])
            print(f"[honeybee] {len(results)}/{len(df)} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
            print(f"           tokens: avg_in={avg_in:.0f}, avg_out={avg_out:.0f}, max_in={token_stats['max_input']} (2048 limit)")

    # Save results
    out_path = os.path.join(output_dir, f"honeybee_{task}_results.csv")
    out_df = pd.DataFrame(results)
    save_csv_safely(out_df, out_path)

    total_time = time.time() - t0
    print(f"[honeybee] saved: {out_path}")
    print(f"[honeybee] total time: {total_time:.1f}s ({total_time/len(df):.2f}s per sample)")

    # Token statistics summary
    avg_input = sum(token_stats["input"]) / len(token_stats["input"])
    avg_output = sum(token_stats["output"]) / len(token_stats["output"])
    max_input = token_stats["max_input"]
    headroom = 2048 - max_input
    print(f"\n[token stats] avg_input={avg_input:.0f}, avg_output={avg_output:.0f}")
    print(f"[token stats] max_input={max_input}/2048 (headroom: {headroom} tokens)")
    print(f"[token stats] max_input sample idx: {token_stats['max_input_idx']}")

    # Run evaluation and save to file
    eval_path = os.path.join(output_dir, f"honeybee_{task}_evaluation.txt")
    if task == "ner":
        eval_results = _evaluate_ner_performance(actual_labels_list, predicted_labels_list, match_type='r')
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write("NER Evaluation Results (Relaxed Match)\n")
            f.write("="*60 + "\n")
            keys = ['target', 'precursor', 'substrate', 'solvent', 'additive']
            for key in keys:
                perf = eval_results['per_class'][key]
                tp, fp, fn = perf['TP'], perf['FP'], perf['FN']
                prec = tp / (tp + fp) if tp + fp > 0 else 0
                rec = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
                f.write(f"{key}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Total={perf['Total']}\n")
            f.write("-"*60 + "\n")
            f.write(f"Accuracy: {eval_results['accuracy']:.3f}\n")
            f.write(f"Macro F1-Score: {eval_results['macro_f1']:.3f}\n")
            f.write(f"Weighted F1-Score: {eval_results['weighted_f1']:.3f}\n")
    else:
        cm = confusion_matrix(actual_labels_list, predicted_labels_list)
        cr = classification_report(actual_labels_list, predicted_labels_list, digits=3)
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report:\n{cr}")
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write(f"HoneyBee Classification Evaluation Results ({task})\n")
            f.write("="*60 + "\n")
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Classification Report:\n{cr}\n")
    print(f"[honeybee] evaluation saved: {eval_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HoneyBee Materials Science LLM Tasks")
    parser.add_argument("--task", required=True, choices=["paragraph", "synthesis-method", "ner"],
                       help="Task type")
    parser.add_argument("--input_csv", required=True, help="Input CSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--preview", default=None, help="Save prompt preview to this path")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization instead of 8-bit")
    args = parser.parse_args()

    cfg = HoneyBeeConfig.create(args.task)
    if args.use_4bit:
        cfg.load_in_4bit = True
        cfg.load_in_8bit = False

    process_dataset(
        input_file=args.input_csv,
        output_dir=args.output_dir,
        task=args.task,
        cfg=cfg,
        preview_prompts_to=args.preview,
    )