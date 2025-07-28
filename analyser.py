import os
import json
import fitz  # PyMuPDF
import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


# Mean pooling helper
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)


# Embedding model
class EmbeddingModel:
    def __init__(self, model_path="distiluse-model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def encode(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return mean_pooling(model_output, encoded_input['attention_mask']).numpy()


def load_input_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    persona = data["persona"]["role"]
    job = data["job_to_be_done"]["task"]
    input_pdfs = [doc["filename"] for doc in data["documents"]]

    return persona, job, input_pdfs


def extract_text_chunks(pdf_dir, pdf_list):
    chunks = []
    for filename in pdf_list:
        path = os.path.join(pdf_dir, filename)
        if not os.path.exists(path):
            continue
        doc = fitz.open(path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text().strip()
            if text:
                chunks.append({
                    "document": filename,
                    "page": page_num + 1,
                    "text": text
                })
        doc.close()
    return chunks


def rank_chunks(persona, job, chunks, model):
    query = f"{persona}: {job}"
    query_embedding = model.encode([query])
    texts = [chunk["text"] for chunk in chunks]
    text_embeddings = model.encode(texts)
    similarities = (query_embedding @ text_embeddings.T)[0]
    for i, score in enumerate(similarities):
        chunks[i]["score"] = float(score)
    ranked_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
    return ranked_chunks


def build_output_json(persona, job, ranked_chunks, top_k=5):
    timestamp = datetime.datetime.now().isoformat()
    top_sections = ranked_chunks[:top_k]
    metadata = {
        "documents": list({chunk["document"] for chunk in ranked_chunks}),
        "persona": persona,
        "job": job,
        "timestamp": timestamp
    }
    extracted_sections = []
    subsection_analysis = []
    for idx, chunk in enumerate(top_sections):
        extracted_sections.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "section_title": chunk["text"].split("\n")[0][:100],
            "importance_rank": idx + 1
        })
        subsection_analysis.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "refined_text": chunk["text"][:500]
        })
    return {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }


def save_output_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    input_root = "/app/input"
    output_root = "/app/output"
    os.makedirs(output_root, exist_ok=True)
    model = EmbeddingModel("/app/distiluse-model")

    for folder in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder)
        if not os.path.isdir(folder_path):
            continue

        input_json_path = os.path.join(folder_path, "challenge1b_input.json")
        pdf_dir = os.path.join(folder_path, "pdfs")

        if not os.path.exists(input_json_path) or not os.path.exists(pdf_dir):
            continue

        print(f"\n🔍 Processing: {folder}")
        persona, job, pdf_list = load_input_config(input_json_path)
        chunks = extract_text_chunks(pdf_dir, pdf_list)
        if not chunks:
            print(f"⚠️ No text found in: {folder}")
            continue

        ranked = rank_chunks(persona, job, chunks, model)
        result = build_output_json(persona, job, ranked)

        output_file = os.path.join(output_root, f"{folder}_output.json")
        save_output_json(result, output_file)
        print(f"✅ Done → {output_file}")


if __name__ == "__main__":
    main()
