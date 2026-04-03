from __future__ import annotations
import os, re, json, time, argparse
from typing import Literal, List, Dict, Any
from collections import defaultdict
import pandas as pd
from openai import OpenAI
from sklearn.metrics import confusion_matrix, classification_report

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
# NER Filtering Functions
# --------------------------------------------------------------------------------------
def _clean_json_string(json_str: str) -> str:
    json_str = json_str.replace("\\xa0", " ").replace("\\", "\\\\")
    json_str = json_str.replace("'", '"')
    json_str = json_str.replace("\n", " ").replace("\r", " ")
    json_str = re.sub(r'\s+', ' ', json_str)
    return json_str


def _extract_ner_entities(content: str) -> List[Dict[str, Any]]:
    """Extract entities from GPT NER output."""
    try:
        content = content.strip()
        entity_pattern = r"\{'(?P<type>[^']+)': '(?P<entity>[^']+)', '(?P<formula>[^']+)', '(?P<metal>[^']+)'\}"
        matches = re.finditer(entity_pattern, content)

        extracted = []
        for match in matches:
            metals = [metal.strip() for metal in match.group("metal").split(",") if metal.strip()] if match.group("metal") else []
            extracted.append({
                "type": match.group("type"),
                "entity": match.group("entity"),
                "formula": match.group("formula"),
                "metals": metals,
            })
        return extracted
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []


def _filter_ner_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
    """Filter entities based on TARGET and PRECURSOR rules."""
    targets = [e for e in entities if e['type'].upper() == 'TARGET']

    if len(targets) != 1 or any("MOF" in target['entity'].upper() for target in targets):
        return None

    target_metals = set(targets[0]['metals'])
    filtered_entities = []
    for entity in entities:
        if entity['type'].upper() == 'TARGET' and not entity['metals']:
            continue
        if entity['type'].upper() == 'PRECURSOR':
            if not entity['metals'] or not set(entity['metals']).intersection(target_metals):
                continue
        filtered_entities.append(entity)
    return filtered_entities


def _parse_filtered_to_labels(filtered_entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert filtered entities to label format for evaluation."""
    parsed_labels = []
    for entity in filtered_entities:
        key = entity['type'].lower()
        if key in ['target', 'precursor', 'substrate', 'solvent', 'additive']:
            parsed_labels.append({key: entity['entity']})
    return parsed_labels


def _extract_actual_ner_labels(label_str: str) -> List[Dict[str, str]]:
    """Extract actual labels from CSV label column."""
    cleaned_item = _clean_json_string(str(label_str).strip())
    if not cleaned_item.startswith("[") and not cleaned_item.endswith("]"):
        cleaned_item = "[" + cleaned_item + "]"
    try:
        labels = json.loads(cleaned_item)
        return labels
    except:
        return []


def _evaluate_ner_performance(actual_labels_list: List, predicted_labels_list: List, match_type: str = 'r') -> Dict[str, Any]:
    """Evaluate NER performance with relaxed or exact matching."""
    keys = ['target', 'precursor', 'substrate', 'solvent', 'additive']
    performance = {key: {'TP': 0, 'FP': 0, 'FN': 0, 'Total': 0} for key in keys}

    for actual, predicted in zip(actual_labels_list, predicted_labels_list):
        if actual is None or predicted is None:
            continue

        actual_dict = defaultdict(set)
        predicted_dict = defaultdict(set)

        for item in actual:
            for key, value in item.items():
                normalized_key = key.lower()
                if normalized_key in keys:
                    actual_dict[normalized_key].add(value.lower())
                    performance[normalized_key]['Total'] += 1

        for item in predicted:
            for key, value in item.items():
                normalized_key = key.lower()
                if normalized_key in keys:
                    predicted_dict[normalized_key].add(value.lower())

        for key in keys:
            tp = 0
            matched_actual = set()
            matched_predicted = set()

            if match_type == 'e':  # Exact match
                tp = len(actual_dict[key] & predicted_dict[key])
                matched_actual = actual_dict[key] & predicted_dict[key]
                matched_predicted = actual_dict[key] & predicted_dict[key]
            elif match_type == 'r':  # Relaxed match
                for act in actual_dict[key]:
                    for pred in predicted_dict[key]:
                        if pred in act or act in pred:
                            tp += 1
                            matched_actual.add(act)
                            matched_predicted.add(pred)

            fp_items = predicted_dict[key] - matched_predicted
            fn_items = actual_dict[key] - matched_actual

            performance[key]['TP'] += tp
            performance[key]['FP'] += len(fp_items)
            performance[key]['FN'] += len(fn_items)

    # Calculate metrics
    total_f1 = 0
    weighted_f1 = 0
    correct_predictions = 0
    total_predictions = 0
    total_labels = sum(performance[key]['Total'] for key in keys)

    print("\n" + "="*60)
    print("NER Evaluation Results (Relaxed Match)" if match_type == 'r' else "NER Evaluation Results (Exact Match)")
    print("="*60)

    for key in performance:
        tp = performance[key]['TP']
        fp = performance[key]['FP']
        fn = performance[key]['FN']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        correct_predictions += tp
        total_predictions += tp + fp + fn
        total_f1 += f1_score / 5
        if total_labels > 0:
            weighted_f1 += f1_score * performance[key]['Total'] / total_labels
        print(f"{key}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}, Total={performance[key]['Total']}")

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print("-"*60)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1-Score: {total_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")
    print("="*60)

    return {
        'accuracy': accuracy,
        'macro_f1': total_f1,
        'weighted_f1': weighted_f1,
        'per_class': performance
    }


# --------------------------------------------------------------------------------------
# System prompts
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
        raise ValueError(f"Unknown task type: {task}")


# --------------------------------------------------------------------------------------
# Few-shot examples (10-shot per class with CoT style)
# --------------------------------------------------------------------------------------
def get_paragraph_examples() -> list:
    """Return 10-shot examples for paragraph classification (synthesis, system, others, performance)."""
    return [
        # === SYNTHESIS (10 examples) ===
        {"role": "user", "content": "PD-Co(OH)2/CC was prepared as follows. In a typical synthesis, the carbon cloth (CC) was cleaned by sonication sequentially in water, ethanol, and acetone for 10 min each and immersed in the 0.1 M Co(NO3)2 solution for electrodeposition. Electrodeposition was carried out by potentiodynamic deposition (PD) in a conventional three-electrode system by a CHI 760E electrochemical analyzer (CH Instruments, Inc.). The carbon cloth (1 cm × 3 cm) was used as the working electrode, saturated calomel electrode (SCE) as the reference electrode, and graphite rod as the counter electrode. The PD process was performed in the potential window 0 to −1.2 V vs SCE at 100 mV s–1 scan rate with 40 cycles. After the deposition, the coated carbon cloth PD-Co(OH)2/CC was rinsed first with water, ethanol and then dried in a vacuum oven at 60 °C for 2 h."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the detailed procedure for preparing PD-Co(OH)2/CC, including the cleaning of carbon cloth, the electrodeposition process, and the post-deposition treatment. This is a typical description of how a material is synthesized in a laboratory setting. Therefore, the answer is synthesis."},
        {"role": "user", "content": "To design efficient S-TMCs-based electrocatalysts, we prepared a spinel structured NiFexCr2-xO4 series (x = 0, 0.2, 0.4, 0.6) using a modified straightforward polymer sol–gel process along with metal precursors and polyacrylonitrile, as indicated in Fig. 1. First, Ni and Cr precursors with Fe precursors as dopants are dissolved in N, N-dimethylformamide under 70 °C for 24 h to form a viscous gel state (Fig. 1a), then the prepared gel compounds were annealed under 700 °C for 3 h (Fig. 1b). The compound was left to cool to room temperature, forming a brown color powder, which is directly used as OER catalysts (Fig. 1c). This NiFexCr2-xO4 synthesis process based on in-situ doping is a facile synthesis strategy compared to classic complicated heteroatom doping in pristine S-TMCs compound."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the preparation process of a series of electrocatalysts, detailing the steps involved in the synthesis, including the sol-gel process, annealing, and the formation of the final product. It does not discuss the system in which the catalysts are used, their performance, or any other unrelated information. Therefore, the answer is synthesis."},
        {"role": "user", "content": "Ternary master alloys ingot with nominal compositions of Mn70Cu30-xNix (x = 0, 1, 3 at%) were prepared by high-frequency induction melting mixture of pure Mn (≥99.99 wt%), pure Cu (≥99.99 wt%) and Ni (≥99.99 wt%) in a quartz tube under Ar atmosphere. The master alloys were remelted in quartz tubes and injected onto a graphite crucible (Φ 14 mm) under argon atmosphere. Subsequently, using the wire-cutting machine, targeted alloy slices (Φ 14 mm × 1 mm) were obtained and further polished to a mirror finish. Then, the alloy slices were dealloyed in a 0.1 M HCl aqueous solution for 24 h to etch Mn until no gas bubble emerged at ambient temperature (25 °C). These dealloyed samples were rinsed repeatedly with ethanol and ultra-purified water, followed by drying at 60 °C for 5 h in vacuum drying oven. The samples after dealloying were named as np-Mn70Cu30-xNix (x = 0, 1, 3 at%)."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the process of creating ternary master alloy ingots, including the melting of pure metals, the cutting and polishing of alloy slices, and the dealloying process in an acidic solution. This is a detailed description of the preparation method of the materials, which falls under the category of 'synthesis' as it outlines the steps taken to synthesize the alloy samples. The answer is synthesis."},
        {"role": "user", "content": "Sodium hypophosphite (NaH2PO2·H2O, 0.3 g), as the P source, was placed upstream of porcelain coat, and NHEMH was placed downstream. Before chemical vapour deposition (CVD), the tube furnace was constantly filled with N2 for 30 min. The heating rate was 5 ℃/min in a 100 mL/min N2 stream. Then, the skeleton was heated at 350 ℃ for 2 h. After heating, the furnace was cooled down to room temperature. To achieve a high-activity bifunctional electrocatalyst, we choose a heating rate of 2 ℃/min, 5 ℃/min, and 10 ℃/min and a NaH2PO2·H2O masses of 0.1 g, 0.2 g, 0.3 g, and 0.4 g as the test parameters."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the process of preparing a bifunctional electrocatalyst using chemical vapor deposition (CVD), including the materials used, the conditions of the process (temperature, heating rate, and atmosphere), and the parameters tested to achieve high activity. It also compares the cost of the resulting material with other commonly used electrode materials. This paragraph is primarily focused on the preparation method and the economic aspect of the materials used in the synthesis. Therefore, the answer is synthesis."},
        {"role": "user", "content": "The NiOOH/Ni(OH)2 was deposited on PIM-CF by using a Savannah S100 ALD reactor (Ultratech Inc.). The sample was loaded in the ALD reaction chamber and heated at 140 °C. The bis(cyclopentadienyl)nickel(II) precursor was preheated to 80 °C, and O3 was used as a counter reactant. A Cambridge NanoTech Savannah Ozone generator was used to produce O3 from pure O2. Dynamic vacuum conditions were used for the uniform coating of NiOOH/Ni(OH)2 on PIM-CF. The pulse, exposure, and purge times for the bis(cyclopentadienyl)nickel(II) precursor were 1, 10, and 10 s, respectively, and for O3 1, 10, and 5 s, respectively. Prior to NiOOH/Ni(OH)2 deposition, PIM-CF were first treated with O3 (with the same conditions for one ALD cycle) to produce −OH functional groups on the surface. 100 cycles of ALD were deposited to acquire Ni@PIM-CF."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the process of depositing NiOOH/Ni(OH)2 on PIM-CF using an ALD reactor, including the conditions and parameters used for the deposition process. It details the precursor used, the temperature settings, the use of ozone, and the number of cycles for the atomic layer deposition. This is a clear description of a material synthesis procedure. Therefore, the answer is synthesis."},
        {"role": "user", "content": "The Co9S8/Cu2S/CF catalyst was prepared through two steps: (i) Co(OH)2 nanosheets were prepared on the Cu2S/CF nanorods (denoted as Co(OH)2/Cu2S/CF) via electrodeposition. In detail, the electrodeposition was performed with Chenhua CHI660E in a three-electrode system in which Cu2S/CF, platinum plated titanium mesh, and saturated calomel electrode (SCE) were used as the working, counter, and reference electrodes, respectively. Co(OH)2 was obtained after electrolytic deposition three times by controlling potential electrolysis at −1.1 V (vs SCE) for 90 s and −0.5 V (vs SCE) for 10 s in 0.1 M Co(NO3)2·6H2O solution. (ii) The Co(OH)2/Cu2S/CF catalyst was immersed in a mixed solution consisting of 30 mL methanol, 30 mL DI water, and 0.05 M Na2S at 25 °C for 15 h. Finally, the Co9S8/Cu2S/CF heterostructure catalyst was obtained."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the detailed process of how the Co9S8/Cu2S/CF catalyst was prepared, including the materials used, conditions applied, and the steps followed to achieve the final product. It clearly outlines the synthesis procedure of the catalyst, from the preparation of intermediate products to the final catalyst formation. Therefore, the answer is synthesis."},
        {"role": "user", "content": "In a typical procedure, Ni(NO3)2·6H2O (4.5 mM), Fe(NO3)3·9H2O (1.5 mM), urea (20.0 mM), and SDS (2.0 mg) were dissolved in 40 mL DI water under magnetic stirring for 10 min at room temperature to form a homogeneous solution. The solution was then transferred to a 50 mL Teflon-lined stainless-steel autoclave containing a piece of CNTs/CP (2 cm × 1 cm) and heated at 120 °C in an oven for 8 h. After the reaction was completed, the system was cooled down to room temperature, and the sample was washed with water and ethanol several times, followed by drying at 60 °C for 12 h."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes a detailed procedure for creating a material, including the specific chemicals used, their concentrations, and the conditions under which they are combined and processed. It clearly outlines the steps taken to synthesize a material, which is characteristic of a synthesis paragraph. Therefore, the answer is synthesis."},
        {"role": "user", "content": "The phosphorization of NiCo MOF/NF was performed via a solid phase reaction with NaH2PO2·H2O as the P source in a tubular furnace. Specifically, 0.15 g of NaH2PO2·H2O was dispersed uniformly at the bottom of the porcelain boat and two pieces of NiCo MOF/NF (1 × 1 cm2) were put on NaH2PO2·H2O side by side, which were covered by another 0.15 g of NaH2PO2·H2O. Then, the porcelain boat was put into a tubular furnace with an atmosphere of Ar. Then, the tubular furnace was heated to 250 °C with a heating rate of 10 °C/min and were kept at that temperature for 2 h. After cooling to room temperature, the phosphorized NiCo MOF/NF was washed with deionized and ethanol and dried at 60 °C in vacuum for 10 h and the sample was named as NiCoP/NF."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the detailed process of how the phosphorization of NiCo MOF/NF was carried out, including the materials used, the conditions of the reaction, and the post-treatment steps. It is focused on the preparation method of the catalyst material. Therefore, the answer is synthesis."},
        {"role": "user", "content": "The composite was prepared using a simple one-step hydrothermal synthesis method (Scheme 2). First, 0.7 g of terephthalic acid and 0.3 g of cobalt acetate were dissolved in a beaker containing 20 mL of ethanol solution, and the solution was marked as A. Second, in another beaker, 0.3 g of manganese acetate was dissolved in 40 mL of DI water, and it was marked as B. Solution B was then poured rapidly into solution A, which was a TPA (terephthalic acid)–ethanolic acid solution under magnetic stirring; without any delay, 1 mL of CND solution was added to the above mixture and allowed to stir for 1 h. The rest of the methodology was the same as the MOF synthesis described above. Then, the precipitate was centrifuged three times with ethanol and distilled water and dried at 60 °C for 6 h."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the process of creating a composite material through a hydrothermal synthesis method. It details the steps taken to dissolve various chemicals, mix solutions, and the subsequent treatment of the precipitate, including centrifugation and drying. Since the focus is on the preparation and creation of the material, the answer is synthesis."},
        {"role": "user", "content": "Graphene oxide (GO) was synthesized using a modified Hummers method.  Typically, 1 g of natural graphite (∼100 mesh) was first heated at 500 °C for 1 h. The metal ions were removed by treating with HCl. The purified graphite was stirred in a mixed solution of H2SO4 (50 mL) and KNO3 (1.2 g), and then 6 g of KMnO4 was slowly added into the suspension. After 6 h, 30 mL of Milli-Q water was added, and the suspension was kept below 80 °C in a cooling bath. After another addition of 200 mL of Milli-Q water and a slow addition of 6 mL of H2O2 (30 wt %), the suspension was stirred for another hour and finally diluted into 1000 mL using water. The suspension was repeatedly decanted for several times until the pH reached 5. Then ultrasonic treatment was used to exfoliate the GO slurry in water."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the process of creating graphene oxide using a modified Hummers method, detailing the specific steps and conditions under which the synthesis occurs. This includes the treatment of graphite, the addition of various chemicals, and the ultrasonic exfoliation to produce GO nanosheets. Therefore, the paragraph is focused on the method of creating a material. The answer is synthesis."},

        # === SYSTEM (10 examples) ===
        {"role": "user", "content": "Following our previously reported method, the PCE was fabricated using polymethylmethacrylate (PMMA) and graphite powder. A viscous solution of 10% PMMA in chloroform was prepared. The graphite powder was dispersed in the PMMA solution at a 60:40 (w/v) ratio. The slurry was sonicated in an ultrasonic bath for 15 min to make a homogeneous solution. The slurry of PMMA and graphite was poured into a glass mold of dimension 10 cm × 10 cm. The polyester sheet (0.1 mm) was used as a bottom of the glass mold and completely sealed with the solution of PMMA to prevent any leakage. The slurry was kept overnight in the mold and covered with pricked paper. The polyester sheet was removed after complete evaporation of the solvent to recover the PCE and cut into a 3 × 8 cm size by a cutter, followed by lamination into the folds of the plastic sheet by heat pressing. The plastic sheets were pre-carved with a 5 mm diameter circle on one side and a 5 × 5 mm window on the other side to be used for the working area and the electrical contacts, respectively. The thickness of the PCE was 350 μm, and conductivity was between 20 and 25 mS/cm."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the detailed process of fabricating a polymer composite electrode (PCE) using PMMA and graphite powder. It outlines the preparation of the solution, the dispersion of graphite, sonication, molding, and the final steps of cutting and lamination. This is a procedural description that details the components and the method of assembly, which is characteristic of a 'system' paragraph type. The answer is system."},
        {"role": "user", "content": "Electrochemical tests were carried out on a Princeton Applied Research Parstat 3000-DX with a three-electrode system in N2-saturated 1.0 M KOH. The Ni(OH)2@1T-MoS2 NWAs or other as-prepared self-supported structure was directly served as working electrode. A graphite rod and a 6.0 M Hg/HgO electrode were served as counter electrode and reference electrode, respectively. All potentials were referenced to the reversible hydrogen electrode (RHE). Linear sweep voltammetry (LSV) measurements were scanned from 0 to −0.5 V (for HER) or 1.3 to 1.8 V (for OER) vs. RHE (negative scanning to eliminate the effect of oxidation peaks for OER) at a rate of 2 mV·s−1 with i-R correction. Tafel slopes were derived from their corresponding LSV data by fitting into the equation: η = a + b·log j."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the experimental setup and the procedures used to carry out electrochemical tests, including the equipment used, the conditions of the tests, and the methods for analyzing the data. It details the components of the three-electrode system, the parameters for the linear sweep voltammetry and cyclic voltammetry, and how the Tafel slopes and electrochemical double-layer capacity are determined. This is characteristic of a 'system' paragraph, as it focuses on the experimental system and methodology. The answer is system."},
        {"role": "user", "content": "Unless otherwise stated, all the powder catalysts were dispersed onto 1 × 1 cm2 carbon paper electrodes (Toray Industries, Inc.). NiO/Ni nanosheets or contrast samples (2 mg) were mixed with 750 μL deionized water, 250 μL ethyl alcohol, 30 μL Nafion® 117 solution, and 6 μL N,N-dimethylformamide (DMF) to form a homogeneous ink, which was then loaded equally onto two pieces of 1 × 1 cm2 carbon paper, such that each one had a loading of 1 mg cm−2. To assess repeatability, these two electrodes were used for reciprocal verification. The as-prepared carbon electrodes were dried in an oven at 70 °C for 60 min to remove the ethanol."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the detailed procedure for preparing catalyst-coated electrodes, including the materials used, their quantities, and the specific steps taken to ensure the catalyst is properly dispersed and the electrodes are prepared for use. This is a description of the experimental setup and the methodology used in the preparation of the catalyst system. Therefore, the answer is system."},
        {"role": "user", "content": "OER current densities were also simultaneously reordered in the same Fe-containing KOH solution. As reference data for comparison, the OER current densities of pristine and Fe-doped thin-film samples were also measured in the same potential range (1.27–1.75 V vs. RHE) in a Fe-free KOH solution. All electrolyte solutions were presaturated by bubbling O2 for 30 min under constant O2 bubbling. The substrate and the connecting copper wire were completely covered with chemically inert insulating epoxy resin after application of silver paint on the back side of a thin-film sample so as to expose the film surface only. Electrochemical impedance spectroscopy to investigate both the uncompensated series resistance (Ru) for iRu correction of the applied potential and the interface charge-transfer resistance was also carried out in the same potentiostat."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the experimental setup and procedures used to measure the oxygen evolution reaction (OER) current densities in Fe-containing and Fe-free KOH solutions. It details the preparation of the samples, the conditions under which the measurements were taken, and the techniques used, such as electrochemical impedance spectroscopy and capacitance measurements. The focus is on the experimental system and methodology, rather than the synthesis of materials or the performance results. Therefore, the answer is system."},
        {"role": "user", "content": "For the deposition of the soft-templated mesoporous support films, different substrates were used. Single-side polished silicon (Si) wafers were obtained from University Wafers with (100) orientation and cleaned with EtOH after a thermal treatment for 2 h at 600 °C in air prior to the deposition of the support films. For the electrical sheet conductivity measurements, quartz glass (SiO2) substrates bought from Science Services GmbH were used and etched prior to film deposition using a mixture of KOH and iPrOH in an ultrasonic bath. Electrochemical measurements were conducted on films deposited on electrically conductive titanium (Ti) substrates, polished with a colloidal silica suspension and subsequently cleaned using a mixture of EtOH and iPrOH (1:1)."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the preparation and treatment of various substrates used for different measurements related to the catalyst. It details the types of substrates, their cleaning and etching processes, and the specific applications for which each substrate is used. This is focused on the setup and preparation of the components of the experimental system rather than the synthesis of the catalyst or its performance. Therefore, the answer is system."},
        {"role": "user", "content": "Samples of the electrocatalyst powder were drop-casted on freshly polished glassy carbon electrodes (GCE; 0.072 cm2, Gamry) for electrochemical measurements that were performed using GCE. For each sample, the preparation of a homogeneous slurry was done by the addition of 2 mg of the catalyst powder to 540 μL of ink solution (400 μL of deionized water (DIW), 100 μL of IPA, and 40 μL of Nafion). The ink was then sonicated in a 2 mL plastic vial through insertion in a water bath for a duration of 30 s under tip sonication."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the detailed procedure for preparing the electrocatalyst samples on different substrates for electrochemical measurements. It includes the preparation of a slurry, sonication, drop-casting, and drying processes, as well as the pretreatment of a nickel foam substrate. These details pertain to the experimental setup and the specific steps taken to prepare the system for testing. Therefore, the answer is system."},
        {"role": "user", "content": "For the two-electrode ASC system, NFPy–CNT was used as the positive electrode material, and reduced graphene oxide (rGO) was used as the negative electrode material. The fabrication procedure of ASC was to mix the electrode material, carbon black, and PVDF in weight ratio of 8:1:1 with NMP and apply mixture to the 1 cm2 of nickel foam for the electrochemical test. After this, poly(vinyl alcohol) (PVA)–KOH electrolyte was prepared as follows: first, 3 g PVA was added in 30 mL DI water and heated to 90 °C under stirring for 2 h, and then 1.5 g KOH was dissolved with DI water. Finally, the above two solutions were mixed under vigorous stirring until a homogeneous solution was obtained."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the assembly process of an asymmetric supercapacitor (ASC) system, detailing the materials used for the electrodes, the preparation of the electrolyte, and the final assembly steps. It does not discuss the synthesis of the materials themselves, nor does it focus on the performance or any other aspects outside of the system construction. Therefore, the answer is system."},
        {"role": "user", "content": "A silicon wafer with 275 nm SiO2 was cleaned and spun with a layer of photoresist (AZ5214, Nalgene). The as-prepared silicon wafer was patterned by ultraviolet lithography and deposited with Ti (20 nm)/Au (50 nm) as metal electrodes through physical vapor deposition (PVD), followed by the transfer of 2D materials and the wet impregnation for single-atom loadings. Then the silicon wafer was spun with a layer of polymethylmethacrylate (495 PMMA A8, Microchem), patterned by e-beam lithography (eBL), and deposited with Ti (20 nm)/Au (50 nm) as the source and drain electrodes, which directly contact the 2D SACs."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the detailed process of preparing a silicon wafer, patterning it, depositing materials, and setting up the electrodes for a micro-device used in electrical measurements. It outlines the steps taken to create the system that will be used for electronic/electrochemical measurement, including the use of a four-electrode microcell. Therefore, the paragraph is focused on the assembly and configuration of the experimental setup. The answer is system."},
        {"role": "user", "content": "Most electrochemical experiments were performed using a Metrohm instrument, Model 797 VA processor. An Autolab potentiostat-galvanostat, Model PGSTAT302, was used to record chronoamperometric curves and electrochemical impedance spectroscopy (EIS) plots. A conventional three-electrode electrochemical cell consisting of a coated or uncoated GCE with a geometric surface area of 0.0314 cm2 as the working electrode and a platinum wire counter electrode and a reference electrode (Ag/AgCl, KCl 3.0 M) was employed to perform all electrochemical measurements."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the experimental setup and the instruments used for electrochemical measurements, including the type of electrochemical cell and the conditions under which the experiments were conducted. It does not discuss the synthesis of materials or the performance results of the experiments. Therefore, the paragraph is focused on the experimental system used for the research. The answer is system."},
        {"role": "user", "content": "The electrochemical performance was measured by using a standard three-electrode system in 1.0 mol L−1 KOH electrolyte (pH = 14) at room temperature (∼25 °C). The as-prepared LDH catalysts, the saturated calomel electrode (SCE), and platinum net were used as the working electrode, reference electrode, and counter electrode, respectively. The Nernst equation was used to calculate the potential of the SCE electrode toward the potential of the reversible hydrogen electrode (RHE)."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the experimental setup and the methods used to measure the electrochemical performance of the LDH catalysts. It details the components of the three-electrode system, the conditions of the electrolyte, the temperature, and the various electrochemical tests conducted. Since it is focused on the experimental setup and the methodology for testing the catalysts, the paragraph should be classified as 'system'. The answer is system."},

        # === OTHERS (10 examples) ===
        {"role": "user", "content": "The kinetic stability of MnOx films at low pH was evaluated by measuring the Mn2+-induced paramagnetic line broadening of the 31P NMR peak for inorganic phosphate. MnOx films exhibit high intrinsic stability in intermediate and alkaline pH regimes; however, the degradation rate of MnOx films increases for more acidic solutions. Figure 4 shows the percent dissolution of MnOx films over time in the presence and absence of applied current (at fixed potential) at very acidic pHs (<2)."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph discusses the stability of MnOx films in different pH conditions and how they are affected by the presence or absence of an applied current. It focuses on the degradation rate and the conditions under which the films are more or less stable. This is related to the chemical stability and durability of the material, rather than its electrical performance. Therefore, the paragraph does not fit the 'performance' category as defined in the instructions. It also does not describe the synthesis of the material or the system in which it is used. Hence, the paragraph is best classified as 'others' because it deals with the chemical stability and degradation behavior of the MnOx films. The answer is others."},
        {"role": "user", "content": "The splitting of water to generate oxygen (oxygen evolution reaction, OER) and hydrogen (hydrogen evolution reaction, HER), H2O → 1/2O2 + H2, holds an ultimate potential to cater the energy demand on a global scale and relax the current environment pollution originating from the mass consumption of fossil sources. However, the electrolysis of water, breaking the O–H bonds and forming O–O double bonds accompanied by the release of protons or electrons, is kinetically sluggish in both acidic and alkaline media, and generally requires a cell potential substantially higher than the thermodynamic value of 1.23 V, that is, a large overpotential."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the context and challenges associated with the water splitting process, including the need for efficient catalysts and the shift towards using nonprecious metals or earth-abundant elements as alternatives to noble metals. It does not describe the synthesis of catalysts, the system setup for water splitting, or the performance of specific catalysts. Therefore, it is more of a background or introduction to the topic rather than fitting into the other categories. The answer is others."},
        {"role": "user", "content": "This implies the presence of V–Fe/Ni scatterings at a distance of around 2.8 Å surrounding V atoms and affords direct evidence for the substitution of V atoms for the Ni sites in Ni(OH)2 lattices. We also made the calculation of the EXAFS spectra by assuming V adsorption on the Ni–Fe LDH layer or occupying the interstitial position. It turns out that in both cases the calculated spectra are quite different from the experimental V K-edge EXAFS spectra of Ni3Fe0.5V0.5 (Supplementary Fig. 8). Furthermore, DFT calculations suggest that V atoms initially placed on the top site of surface Ni or O atoms are relaxed to the interstitial between two LDH layers after structure optimization."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph discusses the analysis of the local environment and positions of vanadium (V) atoms within a nickel-iron hydroxide lattice using extended X-ray absorption fine structure (EXAFS) spectroscopy and density functional theory (DFT) calculations. It compares the stability of different possible positions for V atoms and concludes which position they are most likely to occupy based on the energy calculations and spectral data. This paragraph is focused on the characterization and theoretical understanding of the material's structure, rather than its synthesis, system design, or electrical performance. Therefore, the answer is others."},
        {"role": "user", "content": "After the electrodeposition was carried out in the electrolyte containing Ni(NO3)2·6H2O (0.15 M) and FeSO4·7H2O (0.15 M), many spherical nanoflowers with an average size of about 1–1.5 μm composed of vertically aligned nanosheets were successfully decorated on the surface of NiCoP@NC/NF (Fig. 1c). To further understand the morphology, the TEM and HRTEM images of NiFe LDH/NiCoP@NC that was separated from those of NiFe LDH/NiCoP@NC/NF are shown in Fig. 1d."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes the results of an electrodeposition process, detailing the morphology of the produced nanoflowers and their structural characteristics as observed through TEM and HRTEM imaging. It focuses on the physical characteristics and composition of the materials rather than their synthesis, system setup, or electrical performance. Therefore, the answer is others."},
        {"role": "user", "content": "As one of the emerging 2D nanomaterials, black phosphorus (BP) nanosheets (BP-NSs) possess some unique properties, such as high charge-carrier mobility and tunable bandgap, which are highly desired in applications such as electrode materials or catalysts for energy conversion and storages, including rechargeable batteries and electrocatalysis. Compared with bulk-form BP, 2D BP-NSs provide much more active sites on the ultrathin planar structure, which can dramatically enhance the electrocatalytic activities."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph discusses the unique properties of black phosphorus nanosheets and their applications in energy conversion and storage. It also addresses the challenges associated with their use in oxygen evolution reaction (OER) catalysis and the various strategies proposed to optimize their performance. The focus is on the intrinsic properties of the material and the modifications to improve its catalytic activity, rather than the synthesis or system setup. The paragraph does not exclusively discuss electrical performance but rather the overall electrocatalytic performance, which includes but is not limited to electrical aspects. Therefore, the answer is others."},
        {"role": "user", "content": "To further analyze the chemical composition of as-prepared Mo2C/CoMoS4 samples, XPS was detected as shown in Figure 2. The survey spectrum of Mo2C/CoMoS4 (Figure 2a) showed the characteristic peaks of Co, Mo, O, C, N, and S at 780.52, 232.67, 531.29, 284.32, 398.16, and 168.24 eV, respectively. The presence of N was caused by the unreacted raw materials during the preparation of the 2D composites, while the presence of O might be due to the chemical adsorption of air molecules on the surface of the samples."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the use of X-ray photoelectron spectroscopy (XPS) to analyze the chemical composition of Mo2C/CoMoS4 samples, detailing the detection of various elements and providing explanations for their presence. It focuses on the characterization technique and its findings rather than discussing the synthesis of the material, the system it's part of, or its performance. Therefore, the answer is others."},
        {"role": "user", "content": "The morphologies of the transmetallated products Cu@1 and Cu@3 have been confirmed by field emission scanning electron microscopy (FESEM), transmission electron microscopy (TEM), and high-resolution TEM (HRTEM), and the complete exchange (>99%) of the metal ions has been established by EDX analysis (Figures 4 and 5). It is evidenced from the FESEM analysis that Cu@1 exhibits a flowerlike morphology and Cu@3 exhibits a starlike morphology (Figures 4b and 5b)."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the use of various microscopy and analysis techniques to characterize the morphology, composition, and crystalline structure of transmetallated products. It does not discuss the synthesis of materials, the system they are part of, or their performance in applications. Therefore, the paragraph does not fit into the categories of synthesis, system, or performance but rather describes detailed characterizations of materials. The answer is others."},
        {"role": "user", "content": "To gain deeper insight into the contribution of the constructed interfaces on the Fe-Mo-S/Ni3S2@NF electrode towards HER and OER, DFT calculations of the adsorption free energy (ΔG), charge density difference, electrostatic potential and density of states (DOS) are investigated by building the relevant theoretical models (Fig. 6a). The calculated water adsorption free energy (ΔGH2O) on the reaction surfaces is presented in Fig. 6b."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the use of DFT calculations to investigate the adsorption free energy, charge density difference, electrostatic potential, and density of states for a specific electrode material. It focuses on theoretical models and computational results to understand the material's behavior in hydrogen and oxygen evolution reactions. Given that it deals with theoretical modeling and computational analysis to understand the material's properties and behavior, rather than discussing the synthesis of materials, the setup of a system, or the direct performance of a material in an application, it does not fit neatly into the categories of 'synthesis', 'system', or 'performance'. The answer is others."},
        {"role": "user", "content": "In order to clarify the occupation sites of Fe and V dopants in Ni(OH)2 lattices, we display in Fig. 3 the FT curves of the Fe and V K-edge EXAFS k2χ(k) functions for Ni3Fe, Ni3V, and Ni3Fe0.5V0.5. As references, their Ni K-edge FT curves are also plotted (Fig. 3a). The FT curves of the Fe K-edge data of Ni3Fe and Ni3Fe0.5V0.5 exhibit two prominent coordination peaks at 1.5 and 2.7 Å that are identical to those of their Ni K-edge data (Fig. 3b), suggesting the substitutional doping of Fe in the Ni(OH)2 host."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the use of Fourier Transform (FT) curves and Extended X-ray Absorption Fine Structure (EXAFS) Wavelet Transform (WT) analysis to investigate the substitutional doping of Fe and V in Ni(OH)2 lattices. It focuses on the methodology and analysis of data to understand the structural positioning of dopants within a material. Given the emphasis on analytical techniques and data interpretation to elucidate material structure, the paragraph does not directly discuss the synthesis of materials, the system of the catalyst, or its performance. Therefore, the answer is others."},
        {"role": "user", "content": "The porous structure would enhance significantly the specific surface area and promote the active sites for a catalyst. The high-resolution TEM image (Fig. 1i) indicates a clear crystal lattice fringe with an interplanar spacing of 0.315 nm, corresponding to the (211) plane of CoP3, which indicates that the prepared CoP3 nanoneedle is highly crystalline. Moreover, the selected area electron diffraction (SAED) pattern (Fig. S4†) shows sharp spots which can be indexed to the (110), (200) and (211) planes of CoP3."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph describes the structural characteristics and measurements related to the CoP3 nanoneedles, including their crystallinity, surface area, and pore size, based on various analytical techniques such as TEM, SAED, and BET. It does not directly discuss the synthesis process, the system as a whole, or the performance of the catalyst. Therefore, the answer is others."},

        # === PERFORMANCE (10 examples) ===
        {"role": "user", "content": "In summary, through simple electrodeposition followed by in situ activation, we have successfully fabricated a CS-NiFe0.10Cr0.10 catalyst on a 3D Cu NA substrate for efficient electrochemical OER. The 3D Cu NA nanoarchitecture remarkably facilitates the mass transport process, while the in situ-formed interface metal/metal hydroxide heterojunction significantly promotes the electron transfer procedure. Moreover, Cr3+ incorporation induces electron delocalization around active sites, favoring catalyst redox behavior and improving OER kinetics. Benefiting from the synergy effect of nanostructure construction and electronic structure modulation, the CS-NiFe0.10Cr0.10 shows impressive OER activity: an overpotential of 200 mV at 10 mA/cm2 and Tafel slope of 28 mV/dec in 1 M KOH."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph discusses the fabrication of a catalyst and its impact on the electrochemical oxygen evolution reaction (OER). It mentions the catalyst's architecture and how it facilitates mass transport and electron transfer. The paragraph also provides specific performance metrics, such as overpotential and Tafel slope, which are related to the electrical performance of the catalyst in the OER process. The answer is performance."},
        {"role": "user", "content": "To obtain the most pronounced surface strain condition for the most efficient tunability of OER performance and its fundamental mechanism, the curvature in a flexible membrane would be an excellent strategy, where the surface region suffers the most deformation degree. In this study, we have investigated the strain effect toward OER in flexible van der Waals LNO membrane on the mica substrate. We found that the OER activity is dramatically improved with both compressive and tensile strains exhibiting an ambipolar trend. The current density at 400 mV overpotential can be ∼121% and ∼92% enhanced only by 0.2% compressive and tensile strains via the curvature change of the LNO, compared with the nonstrained one."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph discusses the effects of surface strain on the oxygen evolution reaction (OER) performance of a flexible van der Waals LNO membrane. It mentions how the OER activity is improved under different strain conditions and compares the performance with that of rigid transition metal oxides (TMOs). The focus is on the electrical performance of the material in relation to OER activity, specifically the current density improvements and charge transfer enhancements. Therefore, the answer is performance."},
        {"role": "user", "content": "All of the studies described above indicate that the NiFe LDH/(NiFe)Sx/CMT would be an active and stable bi-functional electrocatalysts for both OER and HER in alkaline solution. Therefore, a two-electrode system was conducted by employing the heterostructure as both anode and cathode electrocatalysts for overall water splitting. It is found that this material affords a current density of 10 mA cm−2 at a low cell voltage of 1.53 V, which is much superior to those of NiFe LDH/(NiFe)Sx, (NiFe)Sx/CMT and NiFe LDH/CMT (Fig. S14), and even outperforms the commercial RuO2||20% Pt/C couple (ca. 1.63 V at 10 mA cm−2)."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph discusses the electrocatalytic activity of a NiFe LDH/(NiFe)Sx/CMT heterostructure for oxygen evolution reaction (OER) and hydrogen evolution reaction (HER) in an alkaline solution. It mentions the current density achieved, the cell voltage, comparison with commercial catalysts, and the long-term stability of the catalyst, all of which are related to the electrical performance of the material in water splitting applications. Therefore, the paragraph should be classified as 'performance'. The answer is performance."},
        {"role": "user", "content": "Inspired by the above discussion, herein, for the first time, we report three-dimensional (3D) hollow Co–Fe–P nanoframes immobilized on N,P-doped carbon nanotubes (CoFeP NFs/NPCNT) as a high-efficiency catalyst toward overall water-splitting. Benefiting from its reasonable nanostructure and composition, the resulting CoFeP NFs/NPCNT displayed remarkable electrocatalytic activities in alkaline electrolyte, with a low overpotential of 278 (or 132) mV at 10 mA cm−2 and small Tafel slope of 39.5 (or 62.9) mV dec−1 for the OER (or HER). More interestingly, CoFeP NFs/NPCNT as both cathodic and anodic catalysts for overall water-splitting just required 1.56 V to deliver 10 mA cm−2 with amazing long-term stability of 60 h."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the electrocatalytic activities of a catalyst material in the context of water-splitting, mentioning specific metrics such as overpotential, Tafel slope, and long-term stability. It focuses on the electrical performance of the catalyst in an application, which is the overall water-splitting process. Therefore, the paragraph is describing the performance of the catalyst, specifically its electrical performance. The answer is performance."},
        {"role": "user", "content": "The Tafel plots derived from LSVs were employed to estimate the OER kinetics. As shown in Figure 3b, the Tafel slope of Ni0.75Fe0.25 BDC was 43.7 mV dec−1, demonstrating most favorable kinetics among all the samples including Ni0.5Fe0.5 BDC (47.4 mV dec−1), RuO2 (57.5 mV dec−1), Ni0.25Fe0.75 BDC (60.2 mV dec−1), Fe BDC (60.4 mV dec−1), and Ni BDC (92.5 mV dec−1). The results also reveal that although the Ni-based MOFs exhibit better OER activity in potentials, the Fe could help to improve the kinetics, which is important for catalytic process."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the use of Tafel plots to estimate the kinetics of the oxygen evolution reaction (OER) for various samples, including nickel-iron metal-organic frameworks (MOFs) and RuO2. It compares the Tafel slopes to assess the kinetics and also mentions electrochemical impedance spectroscopy (EIS) measurements to evaluate charge transfer efficiency. Since the focus is on the electrical properties such as kinetics and charge transfer efficiency, which are directly related to the electrical performance of the catalyst materials, the paragraph should be classified as 'performance'. The answer is performance."},
        {"role": "user", "content": "CoS2–Co(OH)2 HCS was subjected to continuous cyclic voltammetry (CV) scans, in order to monitor the influence of the manner of electrochemical treatment on the performance of the resultant sample in the OER. The first CV scan shows an obvious oxidation peak in positive scan. In the following CV scan, no obvious redox peaks can be found, and different CV curves show little difference. The OER performance of the resultant sample is shown in Figure S11b in the SI), with η 20 = 350 mV and η 100 = 423 mV. The performance is inferior to samples treated by continuous LSV scans."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the results of cyclic voltammetry scans on CoS2–Co(OH)2 HCS and how these results impact the sample's performance in the oxygen evolution reaction (OER). It mentions the observation of oxidation peaks, the lack of significant composition or morphology variation after the first scan, and concludes with specific performance metrics (η 20 = 350 mV and η 100 = 423 mV) compared to samples treated differently. The focus on how the electrochemical treatment influences the OER performance of the sample categorizes this paragraph as discussing the 'performance' of the material. The answer is performance."},
        {"role": "user", "content": "The OER performance of Cu(OH)2/CF, CoNiCu-MOF@Cu2O/CF, and CoNiCu-LDH@CuO/CF was estimated by LSV and compared with that of IrO2/CF. As displayed in Fig. 5a, the polarization curves of CoNiCu-MOF@Cu2O/CF and CoNiCu-LDH@CuO/CF give an oxidation peak (1.20–1.45 V vs. RHE) before oxygen evolution, associated with the conversion of Ni2+ and Co2+ to Ni3+ and Co3+ in alkaline media, respectively. The CoNiCu-LDH@CuO/CF shows high performance compared with Cu(OH)2/CF and CoNiCu-MOF@Cu2O/CF. CoNiCu-LDH@CuO/CF only requires overpotentials of 286 mV and 314 mV, close to that of 285 mV and 302 mV for IrO2/CF, to deliver current densities of 100 and 200 mA cm-2, respectively."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the oxygen evolution reaction (OER) performance of various catalysts by comparing their overpotentials and current densities. It includes comparisons of different materials' abilities to catalyze the reaction, mentioning specific figures and outcomes from experiments. The focus is on how well these materials perform in a given reaction, which is characteristic of a 'performance' paragraph type. Therefore, the answer is performance."},
        {"role": "user", "content": "In addition to the OER efficiency, stability and durability are also critical criteria to evaluate electrocatalysts for real application. Herein, the electrochemical stability of CoFeP NFs/NPCNT was assessed through the normalized chronoamperometric I–T curve for the OER at 1.537 V vs. RHE. As shown in Fig. 5f, there was no noticeable fluctuation for 40 h in the applied current density at the operating potential, exhibiting the excellent OER stability of the catalyst."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the evaluation of the electrocatalyst's efficiency in terms of its stability and durability through specific tests (chronoamperometric I–T curve and CV scanning tests). It presents the outcomes of these tests, such as the lack of fluctuation in current density over time and negligible loss in current density and overpotential after numerous cycles, to demonstrate the catalyst's performance in real application scenarios. Therefore, it focuses on assessing the catalyst's operational effectiveness and longevity, which are aspects of its performance. The answer is performance."},
        {"role": "user", "content": "We further evaluated the catalytic OER performances of the V–Ni3N/NF electrode in alkaline media. As shown in Fig. 5a, the V–Ni3N/NF exhibits high catalytic OER activity, achieving a current density of 10 mA cm−2 at a low potential of 1.519 V, which is smaller than the values for Ni3N/NF (1.547 V), V–Ni3N/NF-1 (1.526 V), V–Ni3N/NF-2 (1.522 V), and RuO2 (1.504 V). The catalytic performance of V–Ni3N/NF exceeds that of RuO2 when the current density is beyond 74 mA cm−2, and it delivers high current densities of 200 and 400 mA cm−2 at low potentials of 1.634 and 1.677 V, respectively."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the evaluation of the catalytic oxygen evolution reaction (OER) performances of the V–Ni3N/NF electrode in alkaline media, comparing its activity and efficiency to other materials and conditions. It provides specific data on current densities achieved at various potentials, comparing these results to those of other materials and configurations. The focus is on the effectiveness and efficiency of the catalytic process under study, making it clear that the paragraph is detailing the outcomes of performance testing of the catalyst in question. Therefore, the answer is performance."},
        {"role": "user", "content": "To investigate the electrode OER kinetic, the OER catalytic activities of these electrocatalysts were also corroborated from Tafel slopes (Figure 5c). As seen, the Tafel slope of 41.2 mV dec−1 was achieved for 3D-1D CoZnP/CNTs nanohybrids, which was 33.3, 38.6, and 84.2 mV dec−1 lower than those of CoZnP NSs (74.5 mV dec−1), Ir/C (79.8 mV dec−1), and CNTs (125.4 mV dec−1) electrodes, respectively, revealing the better OER kinetics of 3D-1D CoZnP/CNTs nanohybrids."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. This paragraph discusses the outcomes of testing the oxygen evolution reaction (OER) kinetics of various electrocatalysts, focusing on their catalytic activities as evidenced by Tafel slopes. It compares the performance of 3D-1D CoZnP/CNTs nanohybrids with other materials, highlighting the superior OER kinetics of the nanohybrids. The paragraph also explains the reasons behind the enhanced electrocatalytic activity, such as increased surface-active sites and modified electronic structures. This detailed examination of the results and their implications for the electrocatalytic process clearly falls under the evaluation of the catalyst's performance in a specific reaction. The answer is performance."},
    ]


def get_synthesis_examples() -> list:
    """Return 5-shot examples (with 2 per class) for synthesis method classification (CoT style)."""
    return [
        # === ELECTRODEPOSITION ===
        {"role": "user", "content": "CoS2 NW loaded on CFP were used as the scaffold for the growth of ultrathin Co(OH)2 NS. The electrodeposition was carried out in a standard three-electrode glass cell at 25 °C, CoS2 NW loaded on CFP as a working electrode, saturated calomel electrode (SCE) as a reference electrode, and a graphite rod as a counter-electrode. The electrolyte is an aqueous solution of Co(NO3)2 (0.05 mol L–1). Ultrathin Co(OH)2 NS was deposited onto CoS2 NW by CV. The CV deposition was conducted in a potential range of −0.5 V to −1.1 V versus SCE with a sweep rate of 10 mV s–1 for different cycles (e.g., 1, 2, 3, and 4 cycles). The resultant sample (CoS2–Co(OH)2 HCS) was rinsed with deionized water. The CoS2–Co(OH)2 HCS was converted to CoOx PHCS by LSV in 1 M KOH solution at a scan rate of 5 mV s–1 for more than four LSV scans."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The paragraph describes a process where ultrathin Co(OH)2 NS is grown on CoS2 NW loaded on CFP using a standard three-electrode glass cell. The process involves using an aqueous solution of Co(NO3)2 as the electrolyte and conducting cyclic voltammetry (CV) deposition in a specified potential range. This clearly indicates that the method used for synthesizing the material is based on applying an electrical current to deposit the material, which is characteristic of the electrodeposition method. Therefore, the answer is electrodeposition."},
        {"role": "user", "content": "Ni foam (NF, thickness: 2 mm, 1*1 cm) was first sonicated in acetone, HCl, water, ethanol several times to remove the surface impurities prior to the electrodepositing, then dried in air at 80℃ for 5 h. The electrodepositing of NiFe composites onto NF was undertaken in a three-electrode system containing equal molar (15 mM) of nickel (II) and iron (III) nitrates at room temperature, using NF as the working electrode, Pt as the counter electrode, Ag/AgCl (saturated KCl solution) as reference electrode. The deposition potential was −1.0 V vs. Ag/AgCl. The best deposition time was determined to be 300 s. After deposition, the obtained NiFe electrocatalyst on NF electrode (denotes as NiFe@NF) was rinsed several times with ethanol, and dried in air overnight."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process described involves cleaning the Ni foam, followed by the use of a three-electrode system to deposit NiFe composites onto the Ni foam using a specific voltage. This clearly indicates the use of an electrical current to induce the deposition of the material onto a substrate, which is characteristic of the electrodeposition method. Therefore, the answer is electrodeposition."},

        # === SOL-GEL ===
        {"role": "user", "content": "The NCFPO/C NPs were synthesized using a sol–gel method according to the procedure reported in our previous study. In the synthesis process, Co(CH3COO)2·4H2O (98%, Sigma-Aldrich), Fe(CH3COO)2 (Fe 29.5%, Alfa Aesar), and citric acid (CA; 99.5%, Sigma-Aldrich) were dissolved in 50 mL of distilled water. Na(CH3COO) (99%, Sigma-Aldrich) and NH4H2PO4 (98%, Sigma-Aldrich) were dissolved in 50 mL of distilled water. The Na/(CoFe)/P/CA molar ratio was 2:1:2:3. The two solutions were then mixed in a three-neck round-bottom flask and were subsequently refluxed at 80 °C for 6 h under continuous stirring. After the reflux, the solution was placed on a hot plate to evaporate the solvent, and the residue was then completely dried in a vacuum oven. The resultant dried gel was heat-treated in a horizon tube furnace at 550 °C for 1 h under the flow of Ar gas."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process described involves dissolving precursors in water, mixing solutions, and then undergoing a reflux and drying process to form a gel. This gel is then heat-treated to obtain the final product. This method is characteristic of the sol-gel process, which involves transitioning from a liquid 'sol' into a solid 'gel' phase. Therefore, the answer is sol-gel."},
        {"role": "user", "content": "To design efficient S-TMCs-based electrocatalysts, we prepared a spinel structured NiFexCr2-xO4 series (x = 0, 0.2, 0.4, 0.6) using a modified straightforward polymer sol–gel process along with metal precursors and polyacrylonitrile, as indicated in Fig. 1. First, Ni and Cr precursors with Fe precursors as dopants are dissolved in N, N-dimethylformamide under 70 °C for 24 h to form a viscous gel state (Fig. 1a), then the prepared gel compounds were annealed under 700 °C for 3 h (Fig. 1b). The compound was left to cool to room temperature, forming a brown color powder, which is directly used as OER catalysts (Fig. 1c). This NiFexCr2-xO4 synthesis process based on in-situ doping is a facile synthesis strategy compared to classic complicated heteroatom doping in pristine S-TMCs compound."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The described process involves dissolving metal precursors in a solvent to form a gel, followed by annealing to obtain the final product. This method is characteristic of the sol-gel process, which involves transitioning from a liquid 'sol' (a colloidal solution) to a solid 'gel' phase. Therefore, the answer is sol-gel."},

        # === SOLID-PHASE ===
        {"role": "user", "content": "A conventional solid state method was used to synthesize the three sample variations of polycrystalline SIZO studied in this work. Powder precursors SrCO3 (Sigma-Aldrich, 99.995%), Ir metal (Alfa Aesar, 99.95%), and ZnO (Alfa Aesar, 99.99%) were used; the ratio of Ir:Zn was varied to make three distinct batches of SIZO with tuned material properties. Stoichiometric strontium precursor was always added at a 1:1 ratio with the total B site precursor. The starting powder reagents were mixed, ground, and preheated in an alumina crucible at 800 °C for 12 h in air. The resulting powders were then ground, pelletized, and calcined in air at 1000 °C for 48 h, before a final grinding, pelletizing, and calcining at 1100 °C for 24 h."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The described method involves mixing, grinding, and heating solid powder precursors, followed by a series of calcination steps. This process does not involve a solvent or a liquid phase, nor does it involve deposition from a vapor or electrochemical deposition. Therefore, given the use of solid precursors and the thermal treatment to induce reaction and crystallization, the synthesis method described is best classified as a solid-phase synthesis. The answer is solid-phase."},
        {"role": "user", "content": "High purity powders of Fe, Co, Ni, Cu, and Mo (≥ 99.5 wt%) of particle sizes of 100–300 mesh were used as the starting material. The powders were mixed in equi-molar (Fe:Co:Ni:Cu:Mo = 1:1:1:1:1) and milled at 250 rpm for 36 h in a planetary ball miller (YXQM-1 L, MITR) under an argon atmosphere. Yttria-stabilized zirconia (YSZ) and stainless steel balls were used as the milling media at a ball-to-powder ratio of 15:1. After that, 0.8 g of mixed powders with or without ball milling was loaded into a stainless-steel mold and pressed at room temperature under a pressure of 600 MPa for 5 min to form a 10×10×1 mm bulk electrode."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process described involves mixing high purity powders of metals, milling them in a planetary ball miller under an argon atmosphere, and then pressing the mixed powders into a bulk form. This method does not involve a liquid phase, vapor phase, or electrochemical deposition, but rather it is a mechanical process that relies on physical forces to mix and compress the powders. Therefore, the answer is solid-phase."},

        # === HYDRO-SOLVOTHERMAL ===
        {"role": "user", "content": "Co/Mn-MOF was synthesized by a solvothermal method reported previously, which is shown in Scheme 1. Briefly, 0.08 g of cobalt(II) acetate tetrahydrate, 0.08 g of manganese acetate, and 0.16 g of BDC were dissolved in 20 mL of DMF, 10 mL of ethanol, and 10 mL of distilled water. The mixture was ultrasonicated for a few minutes and stirred at room temperature. The solution was then kept under continuous magnetic stirring for 2 h till it formed a clear solution. Then, the mixture solution was poured into a 100 mL polymer-coated steel autoclave and kept in an oven at 150 °C for 24 h. The autoclave was allowed to cool completely at room temperature, and the product solution was centrifuged with DMF and ethanol several times and dried in an oven at 90 °C for 12 h."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The synthesis method described involves dissolving precursors in a solvent mixture, ultrasonication, stirring, and then heating the mixture in a sealed autoclave at elevated temperatures. This process is characteristic of the hydrothermal method, but since organic solvents like DMF and ethanol are used in addition to water, it is more accurately described as a solvothermal method. Therefore, the answer is hydro-solvothermal."},
        {"role": "user", "content": "In a typical procedure, Ni(NO3)2·6H2O (4.5 mM), Fe(NO3)3·9H2O (1.5 mM), urea (20.0 mM), and SDS (2.0 mg) were dissolved in 40 mL DI water under magnetic stirring for 10 min at room temperature to form a homogeneous solution. The solution was then transferred to a 50 mL Teflon-lined stainless-steel autoclave containing a piece of CNTs/CP (2 cm × 1 cm) and heated at 120 °C in an oven for 8 h. After the reaction was completed, the system was cooled down to room temperature, and the sample was washed with water and ethanol several times, followed by drying at 60 °C for 12 h."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process described involves dissolving precursors in water, transferring the solution to a Teflon-lined autoclave, and then heating the mixture under pressure. This method is characteristic of hydrothermal synthesis, but since solvents other than water (ethanol for washing) are mentioned, it's more accurate to classify it as hydro-solvothermal synthesis. Therefore, the answer is hydro-solvothermal."},

        # === PRECIPITATION ===
        {"role": "user", "content": "The NFPy was synthesized via a simple co-precipitation method. Initially, Ni(NO3)2·6H2O (4.36 g) and FeCl2·4H2O (1.01 g) were dissolved in distilled water (50 mL) with vigorous stirring for 10 min. Na4P2O7·10H2O (3.35 g) was dissolved in distilled water (50 mL) and stirred for 10 min. Then, the two solutions were mixed with vigorous stirring and kept at room temperature overnight. The bright-green colored precipitates were collected through the centrifuge and washed three times with water. The precipitate was dried using a freeze-dryer and denoted as NFPy."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The described synthesis process involves dissolving precursors in water, mixing them to form a precipitate, and then collecting and drying the precipitate. This method is characterized by the formation of solid particles from a solution, which is the hallmark of the co-precipitation technique. Therefore, the answer is precipitation."},
        {"role": "user", "content": "RuO2 catalyst was prepared according to reported method. K2IrCl6 (0.2 mmol) was added to 50 ml aqueous solution of 0.16 g of sodium hydrogen citrate (0.63 mmol). The red-brown solution was adjusted to pH 7.5 by NaOH (0.25 M) and heated to 95°C with constant stirring. After 30 min, the solution was cooled to room temperature and NaOH solution was added to adjust pH 7.5. The above steps were repeated until the pH was stabilized at 7.5. The colloidal IrOx was precipitated by centrifugation and dried overnight before calcination at 400 °C for 30 min."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process described involves adjusting the pH of a solution to induce the formation of a solid material, which is then separated from the solution by centrifugation. This method of forming a solid phase from a solution through chemical reaction or change in conditions, such as pH adjustment, is characteristic of the precipitation synthesis method. Therefore, the answer is precipitation."},

        # === VAPOR-PHASE ===
        {"role": "user", "content": "CoFe-F was synthesized by low-temperature fluorination of the CoFe precursor. The CoFe precursor was placed at the downstream side of a porcelain boat, and ammonium fluoride (NH4F) was at the upstream side of a porcelain boat. The porcelain boat was put in the center of a tube furnace. The annealing temperature was set at 320 °C for 2 h at a N2 atmosphere with a heating rate of 3 °C min–1 and N2 flow of 10 cc min–1. The final CoFe-F nanosheets were obtained and denoted as CoFe-F-8, CoFe-F-16, and CoFe-F-32, according to the mass ratios of NH4F to CoFe precursor of 8:1, 16:1, and 32:1."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The synthesis method described involves placing a precursor in a tube furnace, heating it under a nitrogen atmosphere, and using ammonium fluoride for fluorination. This process, especially the use of a tube furnace and gas flow, aligns with characteristics of vapor-phase synthesis, where reactions occur in the gas phase at elevated temperatures. Therefore, the answer is vapor-phase."},
        {"role": "user", "content": "The sample of CNTs grown on 3D CP (labeled as CNTs/3D CP) was prepared through chemical vapor deposition method, as described in previous study. Initially, we cut the CP into pieces of 2 cm × 5 cm size. After cleaning by acetone, ethanol, and deionized water for 20 min each under sonication condition, the CP sheet was dipped in HNO3 solution to improve the hydrophilicity of the CP surface, followed by repeated washing with DI water until the CP gained neutral pH. In the second step, the CP was immersed in ethanol solution of Ni(NO3)2 (8 wt%) for 12 h, prior to washing well with DI water, and drying in oven at 60 °C, overnight. In the third step, the Chemical Vapor Deposition (CVD) method was conducted as follows: after placing the pre-treated CP in the CVD machine, temperature was raised to 500 °C for 1 h under H2/Ar atmosphere, followed by increasing the temperature to 700 °C with slightly reducing the H2 flow, and the addition of C2H2 gas; this condition was kept for 3 min."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The process described involves the use of a chemical vapor deposition (CVD) method, which is a type of vapor-phase synthesis. This method involves the deposition of a material from a gas phase onto a substrate through a chemical reaction occurring at the surface of the heated substrate. Given the details provided, such as the use of gases like H2, Ar, and C2H2 at high temperatures to deposit carbon nanotubes (CNTs) onto a substrate, it aligns with the characteristics of vapor-phase synthesis methods. Therefore, the answer is vapor-phase."},

        # === OTHERS ===
        {"role": "user", "content": "The IrO2-Ta2O5 layer was coated on Ti substrates by drop-casting a 2-propanol solution with a composition of Ir:Ta = 75:25 in a mole ratio. Before the coating, a sintered Ti microfiber felt (thickness, 0.1 mm; weight, 15 mg cm–2; porosity, 67%, Nikko Techno, Japan) was ultrasonically cleaned with acetone for 10 min and deionized water for 10 min. The metal sources of the coating solution are tantalum pentachloride (TaCl5; Kojundo Chemical Lab., Japan) and hydrogen hexachloroiridate(IV) n-hydrate (H2IrCl6·nH2O; Ir = 37.4%; Wako Pure Chemical, Japan). The precursor solution of 10 μL was dropped on Ti substrates with a geometrical area of 1 cm2 and dried in an oven at 80 °C for 5 min. After repeating three times, the thermal decomposition of the coated precursor was performed by calcination in air at different temperatures (250–650 °C) for 1 h."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The described process involves dissolving metal precursors in a solvent, drop-casting the solution onto substrates, and then performing a thermal treatment to decompose the precursors into the desired oxide layer. This method does not fit neatly into categories like electrodeposition, sol-gel, solid-phase, hydro-solvothermal, precipitation, or vapor-phase synthesis. It is more aligned with a solution-based coating process followed by thermal decomposition, which is not explicitly listed among the provided synthesis methods. Therefore, the answer is others."},
        {"role": "user", "content": "Graphene oxide (GO) was produced using a modified Hummers method. First, graphite powder (1 g), H2SO4 (27 mL), and H3PO4 (3 mL) were added to a beaker, and the mixture was stirred for 24 h in an ice bath with the temperature being controlled below 20 °C. Then, the beaker was taken out from the ice bath and stirred for another 5 h at 45 °C in a water bath until the solution became sticky. Afterward, deionized water (140 mL) was slowly added into the sticky solution with stirring for 2 h again. After that, H2O2 (20 mL, 30 wt %) was added into the solution with stirring for 30 min to obtain a bright yellow solution. Finally, the solution was centrifuged, washed, and freeze-dried to get the graphene oxide (GO)."},
        {"role": "assistant", "content": "Let's work this out in a step by step way to be sure we have the right answer. The described process involves the chemical modification of graphite to produce graphene oxide using a combination of chemical treatments and mechanical processes such as stirring, centrifugation, and freeze-drying. This method does not fit neatly into categories like electrodeposition, sol-gel, solid-phase, hydro-solvothermal, precipitation, or vapor-phase synthesis. Instead, it is a chemical oxidation process followed by exfoliation, which is more accurately categorized under a broader or different synthesis method category. Therefore, the answer is others."},
    ]


def get_ner_examples() -> list:
    """Return 10-shot examples for NER (5-way entity extraction)."""
    return [
        {"role": "user", "content": "In a typical experiment, RuCl3·xH2O (13.0 mg) and Na2IrCl6·6H2O (40.5 mg) were added to a water/methanol mixed dispersion of ZIF-8 nanocubes (70–100 nm in edge length, Supplementary Fig. 1). After solvothermal treatment (80 °C for 2 h), the RuIrZnOx-U (U stands for unannealed) were obtained (Fig. 1b, Supplementary Fig. 2). RuIrZnOx-U were further annealed at 350 °C for 2 h under 0.5% O2/N2 atmosphere to obtain the RuIrZnOx hollow nanoboxes (h-nanoboxes)."},
        {"role": "assistant", "content": "{'PRECURSOR': 'RuCl3·xH2O', 'RuCl3·xH2O', 'Ru'}, {'PRECURSOR': 'Na2IrCl6·6H2O', 'Na2IrCl6·6H2O', 'Na, Ir'}, {'SOLVENT': 'water', 'H2O', 'None'}, {'SOLVENT': 'methanol', 'CH3OH', 'None'}, {'TARGET': 'RuIrZnOx', 'RuIrZnOx', 'Ru, Ir, Zn'}, {'ADDITIVE': 'ZIF-8 nanocubes', 'C8H12N4Zn', 'Zn'}"},
        {"role": "user", "content": "The precursors for the divalent substituents were poorly crystalline metal peroxides or oxides. Zinc peroxide, ZnO2, was prepared from Zn(NO3)2·6H2O (4.0 g), which was added to a 250 mL polypropylene bottle. The minimal volume of ammonium hydroxide 35% was added under constant stirring, and upon complete dissolution, 10 mL of hydrogen peroxide 30% was added until the solution turned white and opaque."},
        {"role": "assistant", "content": "{'TARGET': 'ZnO2', 'ZnO2', 'Zn'}, {'PRECURSOR': 'Zn(NO3)2·6H2O', 'Zn(NO3)2·6H2O', 'Zn'}, {'ADDITIVE': 'hydrogen peroxide', 'H2O2', 'None'}, {'ADDITIVE': 'ammonium hydroxide', 'NH4OH', 'None'}"},
        {"role": "user", "content": "The Ni6MnO8 microflower was prepared by a simple solvothermal reaction. Primarily, DMF (30 mL) and acetone (30 mL) were mixed as solution A. Ni(NO3)2·6H2O (0.117 g), Mn(NO3)2·4H2O (0.0497 g), and IPA (0.0997 g) were dissolved into solution A and stirred for 90 min to achieve a uniform solution. After the addition of graphene oxide solution (6 mL), the mixed homogeneous solution was added to a Teflon-lined autoclave (100 mL) and held at 160 °C for 4 h."},
        {"role": "assistant", "content": "{'TARGET': 'Ni6MnO8', 'Ni6MnO8', 'Ni, Mn'}, {'SOLVENT': 'DMF', 'C3H7NO', 'None'}, {'SOLVENT': 'acetone', 'C3H6O', 'None'}, {'PRECURSOR': 'Ni(NO3)2·6H2O', 'Ni(NO3)2·6H2O', 'Ni'}, {'PRECURSOR': 'Mn(NO3)2·4H2O', 'Mn(NO3)2·4H2O', 'Mn'}, {'ADDITIVE': 'IPA', 'C3H8O', 'None'}, {'ADDITIVE': 'graphene oxide', '(C2OH)n', 'None'}"},
        {"role": "user", "content": "The bimetallic MOF arrays were synthesized through a one-step solvothermal method. In a typical route for NiFe–MOF, 580 mg of nickel nitrate hexahydrate, 270 mg of ferric chloride hexahydrate, and 166 mg of 1,4-benzenedicarboxylate (BDC) were dispersed in 36 mL ethanol. Then a piece of 2 cm × 2 cm precleaned nickel foam was immersed into the aforementioned dispersion. The vessel was then sealed and maintained at 150 °C for 6 h before cooling naturally to room temperature."},
        {"role": "assistant", "content": "{'TARGET': 'NiFe–MOF', 'NiFe–MOF', 'Ni, Fe'}, {'PRECURSOR': 'nickel nitrate hexahydrate', 'Ni(NO3)2·6H2O', 'Ni'}, {'PRECURSOR': 'ferric chloride hexahydrate', 'FeCl3·6H2O', 'Fe'}, {'ADDITIVE': '1,4-benzenedicarboxylate', 'C8H4O4', 'None'}, {'SOLVENT': 'ethanol', 'CH3CH2OH', 'None'}, {'SUBSTRATE': 'nickel foam', 'Ni', 'Ni'}"},
        {"role": "user", "content": "Prior to synthesis, Ni foam (2 cm × 4 cm × 1 cm) is sonicated in 3 M HCl solution for 15 min to remove the oxide layer attached to the surface, then washed with absolute ethanol and deionized water for three times. The CoMoO4/NF precursor is prepared by a facile hydrothermal process. Firstly, 50 mL of distilled water is used to dissolve 1.45 g Co(NO3)2·6H2O and 0.88 g (NH4)6Mo7O24·4H2O. Meanwhile, 1.2 g CO(NH2)2 is also added as an surfactant. The total solution system is magnetic stirring for 30 min to form a homogeneous solution. Subsequently, transfer the above mixed solution to a 100 mL Teflon-lined autoclave and soak the treated NF into it."},
        {"role": "assistant", "content": "{'SUBSTRATE': 'Ni foam', 'Ni', 'Ni'}, {'TARGET': 'CoMoO4/NF', 'CoMoO4/NF', 'Co, Mo, Ni'}, {'SOLVENT': 'distilled water', 'H2O', 'None'}, {'PRECURSOR': 'Co(NO3)2·6H2O', 'Co(NO3)2·6H2O', 'Co'}, {'PRECURSOR': '(NH4)6Mo7O24·4H2O', '(NH4)6Mo7O24·4H2O', 'Mo'}, {'ADDITIVE': 'CO(NH2)2', 'CO(NH2)2', 'None'}"},
        {"role": "user", "content": "A mixed solution including Co(NO3)2·6H2O (0.20 mmol), Hatz (0.20 mmol), H4bta (0.20 mmol), and 10 mL of the component solvent (Vacetonitrile:VH2O = 1:1) was prepared, poured into a 25 mL autoclave, and heated at 150 °C for 72 h. The reaction mixture was kept for 8 h after the temperature was decreased to 100 °C at a rate of 5 °C h–1 and finally allowed cooled to ambient temperature at a rate of 5 °C h–1; purple areata crystals of 1 were obtained in 60% yield by filtering and washing with water."},
        {"role": "assistant", "content": "{'TARGET': 'purple areata crystals of 1', 'purple areata crystals of 1', 'None'}, {'PRECURSOR': 'Co(NO3)2·6H2O', 'Co(NO3)2·6H2O', 'Co'}, {'ADDITIVE': 'Hatz', 'C8H6N8', 'None'}, {'ADDITIVE': 'H4bta', 'C10H6O8', 'None'}, {'SOLVENT': 'acetonitrile', 'C2H3N', 'None'}, {'SOLVENT': 'H2O', 'H2O', 'None'}"},
        {"role": "user", "content": "In a typical procedure, Ni(NO3)2·6H2O (4.5 mM), Fe(NO3)3·9H2O (1.5 mM), urea (20.0 mM), and SDS (2.0 mg) were dissolved in 40 mL DI water under magnetic stirring for 10 min at room temperature to form a homogeneous solution. The solution was then transferred to a 50 mL Teflon-lined stainless-steel autoclave containing a piece of CNTs/CP (2 cm × 1 cm) and heated at 120 °C in an oven for 8 h. After the reaction was completed, the system was cooled down to room temperature, and the sample was washed with water and ethanol several times, followed by drying at 60 °C for 12 h."},
        {"role": "assistant", "content": "{'PRECURSOR': 'Ni(NO3)2·6H2O', 'Ni(NO3)2·6H2O', 'Ni'}, {'PRECURSOR': 'Fe(NO3)3·9H2O', 'Fe(NO3)3·9H2O', 'Fe'}, {'ADDITIVE': 'urea', 'CH4N2O', 'None'}, {'SOLVENT': 'DI water', 'H2O', 'None'}, {'ADDITIVE': 'SDS', 'NaC12H25SO4', 'Na'}, {'SUBSTRATE': 'CNTs/CP', 'CNTs/CP', 'None'}, {'TARGET': 'Fe-doped Ni(OH)2 NSs', 'FexNi1-x(OH)2', 'Fe, Ni'}"},
        {"role": "user", "content": "NiFeOx/NF: NiFeOx/NF was synthesized by hydrothermal treatment following the method described in the literature. First, 1.96 mmol Ni(NO3)2, 1.88 mmol Fe(NO3)3, and 9.50 mmol CO(NH2)2 were mixed in 152 mL of Milli-Q water (18.2 MΩ cm) under vigorous stirring at room temperature for 10 min. The solution was transferred to a 190 mL Teflon-lined stainless-steel autoclave where the NF substrates were immersed. The sealed autoclave was placed in an oven and heat-treated at 393 K for 12 h."},
        {"role": "assistant", "content": "{'TARGET': 'NiFeOx/NF', 'NiFeOx/NF', 'Ni, Fe'}, {'PRECURSOR': 'Ni(NO3)2', 'Ni(NO3)2', 'Ni'}, {'PRECURSOR': 'Fe(NO3)3', 'Fe(NO3)3', 'Fe'}, {'ADDITIVE': 'CO(NH2)2', 'CO(NH2)2', 'None'}, {'SUBSTRATE': 'NF', 'Ni', 'Ni'}, {'SOLVENT': 'Milli-Q water', 'H2O', 'None'}"},
        {"role": "user", "content": "Typically, 0.3 mmol (NH4)6Mo7O24·4H2O, 12 mmol CH4N2S, and 23 mg NGF were mixed into 70 mL distilled water and stirred for 30 min, then the mixture solution was transferred into a 100 mL Teflon-lined stainless steel autoclave and heated at 180 °C for 12 h to obtain MoS2/NGF precipitate."},
        {"role": "assistant", "content": "{'PRECURSOR': '(NH4)6Mo7O24·4H2O', '(NH4)6Mo7O24·4H2O', 'Mo'}, {'ADDITIVE': 'CH4N2S', 'CH4N2S', 'None'}, {'TARGET': 'MoS2/NGF', 'MoS2/NGF', 'Mo'}, {'SUBSTRATE': 'NGF', 'CxNy', 'None'}, {'SOLVENT': 'distilled water', 'H2O', 'None'}"},
    ]


# --------------------------------------------------------------------------------------
# Message builder
# --------------------------------------------------------------------------------------
def build_messages(task: TaskType, text: str, include_examples: bool) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt(task)}]

    # Add few-shot examples as user-assistant pairs (following the new prompt style)
    if include_examples:
        if task == "paragraph":
            examples = get_paragraph_examples()
            msgs.extend(examples)
        elif task == "synthesis-method":
            examples = get_synthesis_examples()
            msgs.extend(examples)
        elif task == "ner":
            examples = get_ner_examples()
            msgs.extend(examples)

    # Add the actual query
    msgs.append({"role": "user", "content": text})
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
                    cfg: GPTClassificationConfig | Dict[str, Any] = None,
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
      - cfg: GPTClassificationConfig with model/temperature/max_tokens settings
      - fewshot: include example_prompt block (default=True)
      - preview_prompts_to: optional path to dump a human-readable prompt preview (for docs)
      - *_override: optional overrides for model/temperature/max_tokens (takes precedence over cfg)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve model params: cfg → settings → optional overrides
    if isinstance(cfg, GPTClassificationConfig):
        model_name = model_override or cfg.model
        temperature = cfg.temperature if temperature_override is None else temperature_override
        max_tokens = cfg.max_tokens if max_tokens_override is None else max_tokens_override
    else:
        # Legacy dict or None - use settings defaults
        model_name = model_override or settings.GPT_MODEL_NAME
        temperature = settings.GPT_TEMPERATURE if temperature_override is None else temperature_override
        max_tokens = settings.GPT_MAX_TOKENS if max_tokens_override is None else max_tokens_override

    print(f"[gpt] model={model_name}, temperature={temperature}, max_tokens={max_tokens}")

    df = read_csv_safely(input_file, header=None, usecols=[0, 1, 2])
    df.columns = ["id", "label", "text"]

    if preview_prompts_to:
        demo_msgs = build_messages(task, "<example text here>", include_examples=True)
        with open(preview_prompts_to, "w", encoding="utf-8") as f:
            f.write("# Prompt Preview (system + example_prompt + query)\n\n")
            f.write(json.dumps(demo_msgs, ensure_ascii=False, indent=2))

    results = []
    actual_labels_list = []
    predicted_labels_list = []
    t0 = time.time()

    for _, row in df.iterrows():
        qid, text, label = str(row["id"]), str(row["text"]), row["label"]
        msgs = build_messages(task, text, include_examples=fewshot)
        completion = process_text_gpt(msgs, model_name, temperature, max_tokens)

        if task == "ner":
            pred = completion.strip()
            # Extract and filter entities for evaluation
            entities = _extract_ner_entities(pred)
            if entities:
                filtered = _filter_ner_entities(entities)
                if filtered:
                    pred_labels = _parse_filtered_to_labels(filtered)
                else:
                    pred_labels = []
            else:
                pred_labels = []
            predicted_labels_list.append(pred_labels)
            actual_labels_list.append(_extract_actual_ner_labels(label))
        else:
            pred = _extract_answer_token(completion)
            actual_labels_list.append(str(label).strip().lower())
            predicted_labels_list.append(pred)

        results.append({"id": qid, "text": text, "prediction": pred})

        if (len(results) % 20) == 0:
            print(f"[gpt] processed {len(results)}/{len(df)}")

    out_path = os.path.join(output_dir, f"{task}_results.csv")
    save_csv_safely(pd.DataFrame(results), out_path)
    print(f"[gpt] saved: {out_path} (elapsed {time.time()-t0:.1f}s)")

    # Run evaluation and save to file
    eval_path = os.path.join(output_dir, f"{task}_evaluation.txt")
    if task == "ner":
        eval_results = _evaluate_ner_performance(actual_labels_list, predicted_labels_list, match_type='r')
        # Save NER evaluation to file
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write("NER Evaluation Results (Relaxed Match)\n")
            f.write("="*60 + "\n")
            keys = ['target', 'precursor', 'substrate', 'solvent', 'additive']
            for key in keys:
                perf = eval_results['per_class'][key]
                tp, fp, fn = perf['TP'], perf['FP'], perf['FN']
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                f.write(f"{key}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Total={perf['Total']}\n")
            f.write("-"*60 + "\n")
            f.write(f"Accuracy: {eval_results['accuracy']:.3f}\n")
            f.write(f"Macro F1-Score: {eval_results['macro_f1']:.3f}\n")
            f.write(f"Weighted F1-Score: {eval_results['weighted_f1']:.3f}\n")
    else:
        # Classification evaluation (paragraph, synthesis-method)
        cm = confusion_matrix(actual_labels_list, predicted_labels_list)
        cr = classification_report(actual_labels_list, predicted_labels_list, digits=3)
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report:\n{cr}")
        # Save classification evaluation to file
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write(f"Classification Evaluation Results ({task})\n")
            f.write("="*60 + "\n")
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Classification Report:\n{cr}\n")
    print(f"[gpt] evaluation saved: {eval_path}")


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
