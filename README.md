# Domain-aligned LLM framework for trustworthy scientific Q/A via query reformulation retrieval-augmented generation

### Overall workflow
<p align="center">
<img src="https://github.com/gimmeDG/sciqa-qr-rag/blob/main/images/overall%20workflow.jpg" width="90%">
<br>
<em> The domain-aligned Q/A system based on structured database construction and QR-RAG pipeline</em>
</p>
The system transforms unstructured scientific literature into a structured database and integrates it with a QR-RAG pipeline for reliable domain-specific question answering. A total of 11,027 papers are processed through a three-stage NLP pipeline, resulting in a structured database of 2,343 curated papers to power the QR-RAG system.

---

### Preprocess pipeline
<p align="center">
<img src="https://github.com/gimmeDG/sciqa-qr-rag/blob/main/images/Preprocess%20pipeline.jpg?raw=true" width="72%">
<br>
<em>Preprocessing pipeline for water splitting database construction</em>
</p>

Stage 1 - Paragraph Classification: Each paragraph is assigned to one of four classes to identify relevant content.  
Stage 2 - Synthesis Method Classification: Synthesis paragraphs are assigned to one of seven classes for systematic organization by synthesis method.  
Stage 3 - Relational Named Entity Recognition: Five critical entities are extracted, with selective extraction of only those entities that are relationally connected within the paragraph to preserve contextual information.

---

### RAG pipeline
<p align="center">
<img src="https://github.com/gimmeDG/sciqa-qr-rag/blob/main/images/rag%20pipeline.png?raw=true" width="73%">
<br>
<em>C-RAG and QR-RAG pipelines for water splitting domain Q/A</em>
</p>
Based on the structured database, a QR-RAG pipeline is constructed. The pipeline receives user queries and optimizes them through LLM agents (query reformulation, decompose, etc.), then retrieves relevant documents by combining vector and keyword search. Using the retrieved documents as context, it generates final answers, achieving high accuracy (85.6%) compared to conventional C-RAG (21.3%).

---

## User Manual

### Environment

Python 3.10+ recommended
```bash
pip install -r requirements.txt
```

### CLI usage

All tasks are executed through a single entry point:
```bash
python run.py --task <TASK_NAME> [options]
```

#### RAG evaluation
```bash
# GPT-4o backend
python run.py --task rag_json_c_rag_gpt     # JSON + C-RAG (dense retrieval)
python run.py --task rag_json_qr_rag_gpt    # JSON + QR-RAG (hybrid retrieval + query reformulation)
python run.py --task rag_html_c_rag_gpt     # HTML + C-RAG
python run.py --task rag_html_qr_rag_gpt    # HTML + QR-RAG

# LLaMA 3.3-70B backend (via Vertex AI)
python run.py --task rag_json_c_rag_llama
python run.py --task rag_json_qr_rag_llama
python run.py --task rag_html_c_rag_llama
python run.py --task rag_html_qr_rag_llama

# Options
python run.py --task rag_json_qr_rag_gpt --mode descriptive  # RAGAS evaluation
python run.py --task rag_json_qr_rag_gpt --db_variants "123,500,1000"  # Multiple DB sizes
```

#### Interactive mode
For single-query testing without batch evaluation:
```bash
# GPT-4o backend
python run.py --task interactive_gpt

# LLaMA 3.3-70B backend
python run.py --task interactive_llama

# Options
python run.py --task interactive_gpt --format json  # JSON dataset (default)
python run.py --task interactive_gpt --format html  # HTML dataset
python run.py --task interactive_gpt --retrieval c-rag  # C-RAG retrieval (default: qr-rag)
```

#### Other tasks
```bash
# Preprocess - GPT-4 Turbo
python run.py --task gpt_paragraph
python run.py --task gpt_synthesis
python run.py --task gpt_ner

# Preprocess - HoneyBee-7B (Materials Science LLM)
python run.py --task honeybee_paragraph
python run.py --task honeybee_synthesis
python run.py --task honeybee_ner

# Preprocess - LLaMA 3.3-70B (via Vertex AI)
python run.py --task llama_paragraph
python run.py --task llama_synthesis
python run.py --task llama_ner

# MatBERT (fine-tuned)
python run.py --task bert_paragraph
python run.py --task bert_synthesis

# Vector DB
python run.py --task create_vectordb --format both
```
Full task list can be found in `run.py`.

#### MatBERT Setup (Optional)

[MatBERT](https://github.com/lbnlp/MatBERT) is a materials science domain-specific BERT model. To use `bert_*` tasks:

1. Set the model path in `.env`:
   ```
   BERT_MODEL_PATH=ZongqianLi/matbert-base-uncased
   ```
   Or use a local path if downloaded manually:
   ```
   BERT_MODEL_PATH=/path/to/matbert
   ```

Note: The model will be automatically downloaded from [Hugging Face](https://huggingface.co/ZongqianLi/matbert-base-uncased) if not found locally.

#### HoneyBee Model Setup (Optional)

[HoneyBee](https://github.com/BangLab-UdeM-Mila/NLP4MatSci-HoneyBee) is a materials science domain-specific LLM (EMNLP 2023). To use `honeybee_*` tasks:

1. Download model weights from [HoneyBee GitHub](https://github.com/BangLab-UdeM-Mila/NLP4MatSci-HoneyBee)
2. Place the files in the `honeybee/` directory:
   ```
   honeybee/
   ├── llama-7b-hf/    # Base LLaMA model
   └── 7b/             # HoneyBee LoRA weights
   ```
3. Set paths in `.env` (optional, defaults to above structure):
   ```
   HONEYBEE_BASE_MODEL_PATH=honeybee/llama-7b-hf
   HONEYBEE_LORA_PATH=honeybee/7b
   ```

Note: HoneyBee is not required for the main RAG pipeline. GPT or Llama tasks are recommended for general use.

#### LLaMA 3.3-70B Setup (Optional)

LLaMA 3.3-70B can be accessed through various methods (local deployment, cloud APIs, etc.). This project uses [Vertex AI Model Garden MaaS](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama) for managed API access. To use `llama_*` tasks:

1. Set up Google Cloud CLI and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project <YOUR_PROJECT_ID>
   gcloud services enable aiplatform.googleapis.com
   ```

2. Navigate to [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden) and accept the Llama Community License Agreement for the Llama 3.3 model.

3. Set up Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```

4. Configure `.env`:
   ```
   LLAMA_PROJECT_ID=your-gcp-project-id
   LLAMA_LOCATION=us-central1
   LLAMA_MODEL_ID=meta/llama-3.3-70b-instruct-maas
   ```

Note: MaaS (Model as a Service) requires no separate deployment step - Google Cloud manages the endpoint.

---

## Architecture
```bash
sciqa-qr-rag
├─ core/
│  ├─ config.py           # Task-level configurations (BERT, GPT, HoneyBee, LLaMA)
│  ├─ data_utils.py       # Data loading and processing utilities
│  └─ settings.py         # Environment settings and paths
├─ preprocess/
│  ├─ BERT_tasks.py       # MatBERT classification (paragraph, synthesis)
│  ├─ GPT_tasks.py        # GPT-4 classification and NER
│  ├─ HoneyBee_tasks.py   # HoneyBee (Materials Science LLM) tasks
│  ├─ LLaMA_tasks.py      # LLaMA 3.3-70B via Vertex AI
│  └─ build_oer_db.py     # ChromaDB vector database builder
├─ rag/
│  └─ rag_framework.py    # RAG pipeline (C-RAG / QR-RAG)
├─ testset/
│  ├─ paragraph_testset.csv
│  ├─ synthesis_testset.csv
│  ├─ RE-NER_testset.csv
│  ├─ rag_doi_testset.csv
│  ├─ rag_numerical_testset.csv
│  └─ ragas_descriptive_testset.csv
├─ images/
├─ .env.example
├─ requirements.txt
├─ LICENSE
└─ run.py                 # CLI entry point
```
