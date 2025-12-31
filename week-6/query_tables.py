from llmsherpa.readers import LayoutPDFReader
from IPython.core.display import display, HTML
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import Document, ServiceContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Source: https://medium.com/@jitsins/query-complex-pdfs-in-natural-language-with-llmsherpa-ollama-llama3-8b-13b4782243de
# To install:
# 1. run https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
# 2. install and run ollama:
# ollama pull llama3
# ollama run llama3
# 3. Install docker and run:
# docker pull ghcr.io/nlmatics/nlm-ingestor:latest
# docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
# This will expose the api link “http://localhost:5010/api/parseDocument?renderFormat=all” for you to utilize in your code.

from llmsherpa.readers import LayoutPDFReader

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, Document, Settings

from table_cleaner import clean_table_record

import json
from pathlib import Path
from datetime import datetime


llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
pdf_url = "https://s206.q4cdn.com/479360582/files/doc_financials/2024/q1/2024q1-alphabet-earnings-release-pdf.pdf"
#pdf_url = "https://abc.xyz/assets/91/b3/3f9213d14ce3ae27e1038e01a0e0/2024q1-alphabet-earnings-release-pdf.pdf"
Settings.llm = Ollama(model="llama3.2:latest", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(model_name="all-minilm:l12-v2")
Settings.context_window = 4096

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

tables_clean_out = out_dir / "tables_cleaned.jsonl"
qa_out = out_dir / "qa_pairs.json"

pdf_reader = LayoutPDFReader(llmsherpa_api_url)
doc = pdf_reader.read_pdf(pdf_url)

# We'll index ONLY titles
table_docs = []
cleaned_for_save = []

table_id = 0
for section in doc.sections():
    html = section.to_html(include_children=True, recurse=True)
    if "<table" not in html.lower():
        continue

    table_id += 1
    title = (section.title or "").strip() or "(no title)"

    raw_record = {"table_id": table_id, "title": title, "html": html}
    cleaned = clean_table_record(raw_record)
    cleaned_for_save.append(cleaned)

    tsv_blocks = []
    for t in cleaned.get("tables", []):
        tsv = (t.get("tsv") or "").strip()
        if tsv:
            tsv_blocks.append(tsv)

    tables_tsv = "\n\n---\n\n".join(tsv_blocks).strip()
    if not tables_tsv:
        continue

    table_docs.append(
        Document(
            text=title,
            metadata={
                "title": title,
                "table_id": table_id,
                "tables_tsv": tables_tsv,
                "n_tables": len(tsv_blocks),
            },
            excluded_embed_metadata_keys=["tables_tsv"]
        )
    )

print(f"Found {len(cleaned_for_save)} sections containing tables")
print(f"Indexed {len(table_docs)} titles for retrieval")

if not table_docs:
    raise RuntimeError("No tables found.")

with tables_clean_out.open("w", encoding="utf-8") as f:
    for rec in cleaned_for_save:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Saved cleaned tables to: {tables_clean_out}")

index = VectorStoreIndex.from_documents(table_docs)
retriever = index.as_retriever(similarity_top_k=6)

def answer_with_retrieved_tables(question: str) -> str:
    nodes = retriever.retrieve(question)
    if not nodes:
        return "No relevant tables found."

    chunks = []
    for i, node in enumerate(nodes, 1):
        md = node.node.metadata or {}
        title = md.get("title", "(no title)")
        print(title)
        tsv = md.get("tables_tsv", "")
        chunks.append(f"TABLE {i}: {title}\n{tsv}")

    context = "\n\n".join(chunks)

    prompt = (
        "Use ONLY the tables below to answer the question.\n"
        "If the answer is not in the tables, say you can't find it.\n\n"
        f"Question: {question}\n\n"
        f"{context}\n"
    )
    print()
    return Settings.llm.complete(prompt).text

questions = [
    "What was Google's operating margin for 2024?",
    "What % Net income is of the Revenues?",
]

qa_pairs = []
for q in questions:
    ans_text = answer_with_retrieved_tables(q)
    print("\nQ:", q)
    print("A:", ans_text)
    qa_pairs.append({"question": q, "answer": ans_text})

qa_payload = {
    "source_pdf": pdf_url,
    "created_at": datetime.utcnow().isoformat() + "Z",
    "embedding_model": "all-minilm:l12-v2",
    "llm_model": "llama3.2:latest",
    "similarity_top_k": 6,
    "qa_pairs": qa_pairs,
}
with qa_out.open("w", encoding="utf-8") as f:
    json.dump(qa_payload, f, ensure_ascii=False, indent=2)

print(f"Saved Q/A pairs to: {qa_out}")
