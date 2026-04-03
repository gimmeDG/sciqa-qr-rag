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
                           # BERT (train & evaluate with 8:2 split)
                           'bert_paragraph', 'bert_synthesis',
                           # GPT
                           'gpt_paragraph', 'gpt_synthesis', 'gpt_ner',
                           # HoneyBee (Materials Science LLM)
                           'honeybee_paragraph', 'honeybee_synthesis', 'honeybee_ner',
                           # Llama 3.3 70B (via Vertex AI MaaS)
                           'llama_paragraph', 'llama_synthesis', 'llama_ner',
                           # Vector DB
                           'create_vectordb',
                           # RAG - GPT
                           'rag_json_c_rag_gpt', 'rag_html_c_rag_gpt',
                           'rag_json_qr_rag_gpt', 'rag_html_qr_rag_gpt',
                           # RAG - Llama
                           'rag_json_c_rag_llama', 'rag_html_c_rag_llama',
                           'rag_json_qr_rag_llama', 'rag_html_qr_rag_llama',
                           # Interactive RAG (single query)
                           'interactive_gpt', 'interactive_llama',
                       ])
    # Ingestion options
    parser.add_argument('--format', default='both', choices=['json', 'html', 'both'],
                        help='ingestion target for create_vectordb (or dataset format for interactive)')
    # Interactive mode options
    parser.add_argument('--retrieval', default='qr-rag', choices=['c-rag', 'qr-rag'],
                        help='retrieval method for interactive mode')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch size for ChromaDB ingestion')
    # DB variant for batch RAG runs
    parser.add_argument('--db_variants', type=str, default=None,
                        help='Comma-separated DB variants (e.g., "123,300,500,battery")')
    # RAG evaluation mode
    parser.add_argument('--mode', type=str, default='doi', choices=['doi', 'descriptive'],
                        help='RAG evaluation mode: doi (accuracy) or descriptive (RAGAS)')

    args = parser.parse_args()
    TASK = args.task


    # --------------------------------------------------------------------------------------
    # BERT tasks (train & evaluate with 8:2 split)
    # --------------------------------------------------------------------------------------
    if TASK == "bert_paragraph":
        from core.config import BERTClassificationConfig
        from preprocess.BERT_tasks import train_and_evaluate
        config = BERTClassificationConfig.create('paragraph')
        train_and_evaluate(config)

    elif TASK == "bert_synthesis":
        from core.config import BERTClassificationConfig
        from preprocess.BERT_tasks import train_and_evaluate
        config = BERTClassificationConfig.create('synthesis')
        train_and_evaluate(config)


    # --------------------------------------------------------------------------------------
    # GPT classification / NER generation
    # --------------------------------------------------------------------------------------
    elif TASK == "gpt_paragraph":
        from preprocess.GPT_tasks import process_dataset
        from core.config import GPTClassificationConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = GPTClassificationConfig.create('paragraph')
        input_file = os.path.join(settings.DATA_DIR, "paragraph_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "gpt_paragraph")
        process_dataset(input_file, output_dir, 'paragraph', cfg)

    elif TASK == "gpt_synthesis":
        from preprocess.GPT_tasks import process_dataset
        from core.config import GPTClassificationConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = GPTClassificationConfig.create('synthesis')
        input_file = os.path.join(settings.DATA_DIR, "synthesis_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "gpt_synthesis")
        process_dataset(input_file, output_dir, 'synthesis-method', cfg)

    elif TASK == "gpt_ner":
        from preprocess.GPT_tasks import process_dataset
        from core.config import GPTClassificationConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = GPTClassificationConfig.create('ner')
        input_file = os.path.join(settings.DATA_DIR, "RE-NER_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "gpt_ner")
        process_dataset(input_file, output_dir, 'ner', cfg)


    # --------------------------------------------------------------------------------------
    # HoneyBee (Materials Science LLM) tasks
    # --------------------------------------------------------------------------------------
    elif TASK == "honeybee_paragraph":
        from preprocess.HoneyBee_tasks import process_dataset
        from core.config import HoneyBeeConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = HoneyBeeConfig.create('paragraph')
        input_file = os.path.join(settings.DATA_DIR, "paragraph_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "honeybee_paragraph")
        process_dataset(input_file, output_dir, 'paragraph', cfg)

    elif TASK == "honeybee_synthesis":
        from preprocess.HoneyBee_tasks import process_dataset
        from core.config import HoneyBeeConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = HoneyBeeConfig.create('synthesis-method')
        input_file = os.path.join(settings.DATA_DIR, "synthesis_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "honeybee_synthesis")
        process_dataset(input_file, output_dir, 'synthesis-method', cfg)

    elif TASK == "honeybee_ner":
        from preprocess.HoneyBee_tasks import process_dataset
        from core.config import HoneyBeeConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = HoneyBeeConfig.create('ner')
        input_file = os.path.join(settings.DATA_DIR, "RE-NER_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "honeybee_ner")
        process_dataset(input_file, output_dir, 'ner', cfg)


    # --------------------------------------------------------------------------------------
    # Llama 3.3 70B (via Vertex AI MaaS) tasks
    # --------------------------------------------------------------------------------------
    elif TASK == "llama_paragraph":
        from preprocess.LLaMA_tasks import process_dataset
        from core.config import LlamaConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = LlamaConfig.create('paragraph')
        input_file = os.path.join(settings.DATA_DIR, "paragraph_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "llama_paragraph")
        process_dataset(input_file, output_dir, 'paragraph', cfg)

    elif TASK == "llama_synthesis":
        from preprocess.LLaMA_tasks import process_dataset
        from core.config import LlamaConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = LlamaConfig.create('synthesis-method')
        input_file = os.path.join(settings.DATA_DIR, "synthesis_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "llama_synthesis")
        process_dataset(input_file, output_dir, 'synthesis-method', cfg)

    elif TASK == "llama_ner":
        from preprocess.LLaMA_tasks import process_dataset
        from core.config import LlamaConfig
        from core import settings
        from core.data_utils import create_timestamped_output_dir
        cfg = LlamaConfig.create('ner')
        input_file = os.path.join(settings.DATA_DIR, "RE-NER_testset.csv")
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, "llama_ner")
        process_dataset(input_file, output_dir, 'ner', cfg)


    # --------------------------------------------------------------------------------------
    # Vector DB / RAG
    # --------------------------------------------------------------------------------------
    elif TASK == "create_vectordb":
        from preprocess.build_oer_db import main as db_main
        sys.argv = [sys.argv[0], "--format", args.format, "--batch_size", str(args.batch_size)]
        db_main()

    elif TASK in ("rag_json_c_rag_gpt", "rag_html_c_rag_gpt", "rag_json_qr_rag_gpt", "rag_html_qr_rag_gpt",
                   "rag_json_c_rag_llama", "rag_html_c_rag_llama", "rag_json_qr_rag_llama", "rag_html_qr_rag_llama"):
        import importlib
        import core.settings as settings_module
        from core.data_utils import create_timestamped_output_dir
        import rag.rag_framework as rag_module

        # Set LLM backend based on task name
        if "_llama" in TASK:
            rag_module.set_llm_backend("llama")
            base_task = TASK.replace("_llama", "")
        else:
            rag_module.set_llm_backend("openai")
            base_task = TASK.replace("_gpt", "")

        fmt = "json" if "json" in base_task else "html"
        retrieval_mode = "qr-rag" if "_qr_rag" in base_task else "c-rag"
        eval_mode = args.mode  # "doi" or "descriptive" (RAGAS)

        # Select CSV based on evaluation mode
        if eval_mode == "descriptive":
            csv_path = str(settings_module.RAG_RAGAS_CSV_PATH)
        else:
            csv_path = None  # Use default from settings (RAG_QA_CSV_PATH)

        # Handle multiple DB variants if specified
        if args.db_variants:
            variants = [v.strip() for v in args.db_variants.split(",")]
            for variant in variants:
                print("\n" + "=" * 70)
                print(f"  Running with DB variant: {variant}")
                print("=" * 70)
                # Override environment variables for this variant
                # Special handling for "battery" and "batteryonly" variants
                if variant == "battery":
                    if fmt == "json":
                        os.environ["RAG_JSON_COLLECTION"] = "oer_battery_json"
                        os.environ["RAG_JSON_PERSIST_DIR"] = "./.chroma_json_battery"
                    else:
                        os.environ["RAG_HTML_COLLECTION"] = "oer_battery_html"
                        os.environ["RAG_HTML_PERSIST_DIR"] = "./.chroma_html_battery"
                elif variant == "batteryonly":
                    if fmt == "json":
                        os.environ["RAG_JSON_COLLECTION"] = "oer_battery_json"
                        os.environ["RAG_JSON_PERSIST_DIR"] = "./.chroma_json_batteryonly"
                    else:
                        os.environ["RAG_HTML_COLLECTION"] = "oer_battery_html"
                        os.environ["RAG_HTML_PERSIST_DIR"] = "./.chroma_html_batteryonly"
                else:
                    if fmt == "json":
                        os.environ["RAG_JSON_COLLECTION"] = f"sciqa_json_{variant}"
                        os.environ["RAG_JSON_PERSIST_DIR"] = f"./.chroma_json_{variant}"
                    else:
                        os.environ["RAG_HTML_COLLECTION"] = f"sciqa_html_{variant}"
                        os.environ["RAG_HTML_PERSIST_DIR"] = f"./.chroma_html_{variant}"
                # Reload BOTH settings AND rag_framework to pick up new env vars
                importlib.reload(settings_module)
                importlib.reload(rag_module)
                # Re-apply LLM backend after reload (reload resets to default "openai")
                if "_llama" in TASK:
                    rag_module.set_llm_backend("llama")
                else:
                    rag_module.set_llm_backend("openai")
                # Create RAG instance with updated settings
                rag = rag_module.UnifiedRAG(dataset_format=fmt, retrieval=retrieval_mode, csv_path=csv_path)
                # Use "ragas_" prefix for descriptive mode output folders
                task_prefix = "ragas_" if eval_mode == "descriptive" else ""
                output_dir = create_timestamped_output_dir(settings_module.RESULTS_DIR, f"{task_prefix}{TASK}_{variant}")
                rag.evaluate_csv(save_dir=output_dir, eval_mode=eval_mode)
        else:
            rag = rag_module.UnifiedRAG(dataset_format=fmt, retrieval=retrieval_mode, csv_path=csv_path)
            # Use "ragas_" prefix for descriptive mode output folders
            task_prefix = "ragas_" if eval_mode == "descriptive" else ""
            output_dir = create_timestamped_output_dir(settings_module.RESULTS_DIR, f"{task_prefix}{TASK}")
            rag.evaluate_csv(save_dir=output_dir, eval_mode=eval_mode)

    # --------------------------------------------------------------------------------------
    # Interactive RAG (single query mode)
    # --------------------------------------------------------------------------------------
    elif TASK in ("interactive_gpt", "interactive_llama"):
        import json
        import time
        import rag.rag_framework as rag_module
        from core import settings
        from core.data_utils import create_timestamped_output_dir

        # Set LLM backend
        llm_backend = "llama" if "_llama" in TASK else "openai"
        if "_llama" in TASK:
            rag_module.set_llm_backend("llama")
        else:
            rag_module.set_llm_backend("openai")

        # Use json format by default, or user-specified
        fmt = "json" if args.format == "both" else args.format
        retrieval = args.retrieval

        # Create output directory for this session
        output_dir = create_timestamped_output_dir(settings.RESULTS_DIR, TASK)
        session_start = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"interactive_{fmt}_{retrieval}_{session_start}.json")

        # Initialize session data
        session_data = {
            "metadata": {
                "session_start": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task": TASK,
                "dataset_format": fmt,
                "retrieval_method": retrieval,
                "llm_backend": llm_backend,
            },
            "queries": []
        }

        print("\n" + "=" * 70)
        print("  Interactive RAG Mode")
        print(f"  Format: {fmt} | Retrieval: {retrieval} | LLM: {'Llama' if '_llama' in TASK else 'GPT'}")
        print(f"  Results will be saved to: {results_file}")
        print("=" * 70)
        print("  Type your question and press Enter.")
        print("  To exit: type 'exit', 'quit', 'q', or press Ctrl+C\n")

        rag = rag_module.UnifiedRAG(dataset_format=fmt, retrieval=retrieval)
        query_count = 0

        while True:
            try:
                query = input("Query: ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit", "q"):
                    break

                query_count += 1
                query_start = time.time()
                print("\nSearching...\n")
                response, retrieved, _ = rag._runner.answer(query, top_k=5)
                execution_time = round(time.time() - query_start, 2)

                print("-" * 70)
                print("Answer:")
                print(response)
                print("-" * 70)
                print(f"(Retrieved documents: {len(retrieved)}) | Time: {execution_time}s\n")

                # Format retrieval info (similar to other modes)
                retrieval_info = []
                for rank, para in enumerate(retrieved, 1):
                    retrieval_info.append({
                        "rank": rank,
                        "doc_id": para.get("doc_id", ""),
                        "similarity_score": round(para.get("score", 0.0), 4),
                        "section": para.get("section", ""),
                        "paragraph_idx": para.get("paragraph_idx", -1),
                        "text": para.get("text", ""),
                        "text_length": len(para.get("text", "")),
                    })

                # Save query result
                query_result = {
                    "query_number": query_count,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "question": query,
                    "response": response,
                    "retrieval": {
                        "top_k_paragraphs": retrieval_info,
                        "unique_documents": list({p.get("doc_id") for p in retrieved if p.get("doc_id")}),
                        "num_unique_documents": len({p.get("doc_id") for p in retrieved if p.get("doc_id")}),
                    },
                    "execution_time_sec": execution_time,
                }
                session_data["queries"].append(query_result)

                # Save after each query (to prevent data loss)
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)

            except KeyboardInterrupt:
                print("\n")
                break

        # Finalize session
        session_data["metadata"]["session_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        session_data["metadata"]["total_queries"] = query_count

        # Final save
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"  Session ended. Total queries: {query_count}")
        print(f"  Results saved to: {results_file}")
        print("=" * 70)

        # Skip "Task completed" message for interactive mode
        sys.exit(0)

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
