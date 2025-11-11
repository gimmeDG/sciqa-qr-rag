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
<img src="https://github.com/gimmeDG/sciqa-qr-rag/blob/main/images/preprocess%20pipeline.jpg" width="72%">
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
python run.py --task rag_json_c_rag     # JSON + C-RAG (dense retrieval)
python run.py --task rag_json_qr_rag    # JSON + QR-RAG (hybrid retrieval + query reformulation)
python run.py --task rag_html_c_rag     # HTML + C-RAG
python run.py --task rag_html_qr_rag    # HTML + QR-RAG
```

#### Other tasks
```bash
# Preprocess
python run.py --task bert_paragraph_train
python run.py --task bert_paragraph_classify
python run.py --task gpt_paragraph
python run.py --task gpt_ner

# Vector DB
python run.py --task create_vectordb --format both

# Evaluation
python run.py --task bert_paragraph_eval
python run.py --task gpt_paragraph_eval
```
Full task list can be found in `run.py`.

---

## Architecture
```bash
sciqa-qr-rag
├─ core/
│ ├─ settings.py
│ ├─ config.py
│ └─ data_utils.py
├─ preprocess/
│ ├─ BERT_tasks.py
│ ├─ GPT_tasks.py
│ ├─ GPT_ner_filter.py
│ ├─ evaluation.py
│ └─ build_chromadb.py
├─ rag/
│ └─ rag_framework.py
├─ data_example/
│ ├─ paragraph_raw.csv
│ ├─ paragraph_gold.csv
│ ├─ synthesis_raw.csv
│ ├─ synthesis_gold.csv
│ ├─ ner_raw.csv
│ ├─ ner_gold.csv
│ └─ qa_testset.csv
├─ images/
├─ .env.example
├─ requirements.txt
├─ LICENSE
└─ run.py
```
