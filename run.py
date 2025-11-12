from __future__ import annotations
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='OER Text Classification & RAG Pipeline')
    parser.add_argument('--task', type=str, required=True,
                       choices=[
                           # BERT
                           'bert_paragraph_train', 'bert_paragraph_classify',
                           'bert_synthesis_train', 'bert_synthesis_classify',
                           'full_bert_pipeline',
                           # GPT
                           'gpt_paragraph', 'gpt_synthesis', 'gpt_ner',
                           # NER (pipeline)
                           'gpt_ner_filter', 'gpt_ner_eval',
                           # Vector DB
                           'create_vectordb',
                           # RAG (4 modes)
                           'rag_json_c_rag', 'rag_html_c_rag',
                           'rag_json_qr_rag', 'rag_html_qr_rag',
                           # EVALUATION (classification)
                           'bert_paragraph_eval', 'bert_synthesis_eval',
                           'gpt_paragraph_eval', 'gpt_synthesis_eval'
                       ])
    # Optional overrides for evaluation
    parser.add_argument('--pred_csv', default=None, help='prediction csv path (override)')
    parser.add_argument('--gold_csv', default=None, help='gold csv path (override)')
    parser.add_argument('--save_json', default=None, help='metrics json output path')
    parser.add_argument('--match_type', default='r', choices=['e', 'r'],
                        help="NER matching: 'e' (exact) or 'r' (relaxed)")
    # Ingestion options
    parser.add_argument('--format', default='both', choices=['json', 'html', 'both'],
                        help='ingestion target for create_vectordb')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch size for ChromaDB ingestion')

    args = parser.parse_args()
    TASK = args.task


    # --------------------------------------------------------------------------------------
    # BERT tasks
    # --------------------------------------------------------------------------------------
    if TASK == "bert_paragraph_train":
        from core.config import BERTClassificationConfig
        from preprocess.BERT_tasks import train_bert_classifier
        config = BERTClassificationConfig.create('paragraph')
        train_bert_classifier(config)

    elif TASK == "bert_paragraph_classify":
        from core.config import BERTClassificationConfig
        from preprocess.BERT_tasks import classify_text
        config = BERTClassificationConfig.create('paragraph')
        classify_text(config)

    elif TASK == "bert_synthesis_train":
        from core.config import BERTClassificationConfig
        from preprocess.BERT_tasks import train_bert_classifier
        config = BERTClassificationConfig.create('synthesis')
        train_bert_classifier(config)

    elif TASK == "bert_synthesis_classify":
        from core.config import BERTClassificationConfig
        from preprocess.BERT_tasks import classify_text
        config = BERTClassificationConfig.create('synthesis')
        classify_text(config)

    elif TASK == "full_bert_pipeline":
        from core.config import BERTClassificationConfig
        from preprocess.BERT_tasks import train_bert_classifier, classify_text
        print("\n[1/4] Training paragraph classifier.")
        config_para = BERTClassificationConfig.create('paragraph')
        train_bert_classifier(config_para)
        print("\n[2/4] Classifying paragraphs.")
        classify_text(config_para)
        print("\n[3/4] Training synthesis classifier.")
        config_syn = BERTClassificationConfig.create('synthesis')
        train_bert_classifier(config_syn)
        print("\n[4/4] Classifying synthesis.")
        classify_text(config_syn)


    # --------------------------------------------------------------------------------------
    # GPT classification / NER generation
    # --------------------------------------------------------------------------------------
    elif TASK == "gpt_paragraph":
        from preprocess.GPT_tasks import process_dataset
        from core.config import GPTClassificationConfig
        from core import settings
        cfg = GPTClassificationConfig.create('paragraph')
        # Input CSV expected at: DATA_DIR/paragraph_raw.csv  (id,text,label; label can be dummy)
        input_file = os.path.join(settings.DATA_DIR, "paragraph_raw.csv")
        output_dir = os.path.join(settings.RESULTS_DIR, "GPT/gpt_paragraph")
        process_dataset(input_file, output_dir, 'paragraph', cfg)

    elif TASK == "gpt_synthesis":
        from preprocess.GPT_tasks import process_dataset
        from core.config import GPTClassificationConfig
        from core import settings
        cfg = GPTClassificationConfig.create('synthesis')
        # Input CSV expected at: DATA_DIR/synthesis_raw.csv  (id,text,label; label can be dummy)
        input_file = os.path.join(settings.DATA_DIR, "synthesis_raw.csv")
        output_dir = os.path.join(settings.RESULTS_DIR, "GPT/gpt_synthesis")
        process_dataset(input_file, output_dir, 'synthesis-method', cfg)

    elif TASK == "gpt_ner":
        from preprocess.GPT_tasks import process_dataset
        from core import settings
        cfg = {"model": "gpt-4", "temperature": 0, "max_tokens": 1000}
        # Input CSV expected at: DATA_DIR/ner_raw.csv  (id,text,label; label can be dummy)
        input_file = os.path.join(settings.DATA_DIR, "ner_raw.csv")
        output_dir = os.path.join(settings.RESULTS_DIR, "GPT/synthesis_ner/ner_output")
        process_dataset(input_file, output_dir, 'ner', cfg)


    # --------------------------------------------------------------------------------------
    # NER post-processing / evaluation
    # --------------------------------------------------------------------------------------
    elif TASK == "gpt_ner_filter":
        from preprocess.GPT_ner_filter import filter_ner_results
        from core import settings
        inp = os.path.join(settings.RESULTS_DIR, "GPT/synthesis_ner/ner_output/ner_results_raw.csv")
        out = os.path.join(settings.RESULTS_DIR, "GPT/synthesis_ner/ner_output/ner_results_filtered.csv")
        filter_ner_results(inp, out)

    elif TASK == "gpt_ner_eval":
        from preprocess.evaluation import evaluate_ner
        from core import settings
        default_pred = os.path.join(settings.RESULTS_DIR, "GPT/synthesis_ner/ner_output/ner_results_filtered.csv")
        default_gold = os.path.join(settings.DATA_DIR, "ner_gold.csv")
        default_out = os.path.join(settings.RESULTS_DIR, "metrics/ner_eval.json")

        pred_csv = args.pred_csv or default_pred
        gold_csv = args.gold_csv or default_gold
        save_json = args.save_json or default_out
        match_type = args.match_type or "r"

        print(f"[ner-eval] pred={pred_csv}")
        print(f"[ner-eval] gold={default_gold}")
        print(f"[ner-eval] match_type={match_type}")
        evaluate_ner(pred_csv=pred_csv, gold_csv=gold_csv,
                     match_type=match_type, save_json=save_json)


    # --------------------------------------------------------------------------------------
    # Vector DB / RAG
    # --------------------------------------------------------------------------------------
    elif TASK == "create_vectordb":
        from preprocess.build_chromadb import main as db_main
        sys.argv = [sys.argv[0], "--format", args.format, "--batch_size", str(args.batch_size)]
        db_main()

    elif TASK in ("rag_json_c_rag", "rag_html_c_rag", "rag_json_qr_rag", "rag_html_qr_rag"):
        from rag.rag_framework import UnifiedRAG
        from core import settings
        fmt = "json" if "json" in TASK else "html"
        mode = "qr-rag" if "_qr_rag" in TASK else "c-rag"
        rag = UnifiedRAG(dataset_format=fmt, retrieval=mode)
        rag.evaluate_csv(save_dir=os.path.join(settings.RESULTS_DIR, "RAG"))


    # --------------------------------------------------------------------------------------
    # Classification evaluation (BERT/GPT)
    # --------------------------------------------------------------------------------------
    elif TASK in ("bert_paragraph_eval", "bert_synthesis_eval",
                  "gpt_paragraph_eval", "gpt_synthesis_eval"):
        from preprocess.evaluation import evaluate_classification
        from core import settings

        if TASK == "bert_paragraph_eval":
            default_pred = os.path.join(settings.RESULTS_DIR, "BERT/paragraph/prediction.csv")
            default_gold = os.path.join(settings.DATA_DIR, "paragraph_gold.csv")
            labels = ["synthesis", "system", "performance", "others"]
        elif TASK == "bert_synthesis_eval":
            default_pred = os.path.join(settings.RESULTS_DIR, "BERT/synthesis/prediction.csv")
            default_gold = os.path.join(settings.DATA_DIR, "synthesis_gold.csv")
            labels = [
                "electrodeposition",
                "sol-gel",
                "solid-phase",
                "hydro-solvothermal",
                "precipitation",
                "vapor-phase",
                "others",
            ]
        elif TASK == "gpt_paragraph_eval":
            default_pred = os.path.join(settings.RESULTS_DIR, "GPT/gpt_paragraph/paragraph_results.csv")
            default_gold = os.path.join(settings.DATA_DIR, "paragraph_gold.csv")
            labels = ["synthesis", "system", "performance", "others"]
        elif TASK == "gpt_synthesis_eval":
            default_pred = os.path.join(settings.RESULTS_DIR, "GPT/gpt_synthesis/synthesis-method_results.csv")
            default_gold = os.path.join(settings.DATA_DIR, "synthesis_gold.csv")
            labels = [
                "electrodeposition",
                "sol-gel",
                "solid-phase",
                "hydro-solvothermal",
                "precipitation",
                "vapor-phase",
                "others",
            ]

        pred_csv = args.pred_csv or default_pred
        gold_csv = args.gold_csv or default_gold
        save_json = args.save_json or os.path.join(settings.RESULTS_DIR, "metrics", f"{TASK}.json")

        print(f"[eval] pred={pred_csv}")
        print(f"[eval] gold={gold_csv}")
        evaluate_classification(
            pred_csv=pred_csv,
            gold_csv=gold_csv,
            id_col="id",
            pred_col="prediction",
            gold_col="label",
            label_order=labels,
            save_json=save_json
        )

    print("\n" + "=" * 70)
    print("Task completed")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
