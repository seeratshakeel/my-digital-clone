import json
import hashlib
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# Project folder
BASE_DIR = Path(__file__).parent

# Retrieval settings
CHUNK_SIZE = 250
OVERLAP = 40
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Files where index data will be saved
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"
CHUNKS_FILE = BASE_DIR / "chunks.json"
META_FILE = BASE_DIR / "index_meta.json"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_documents():
    documents = []

    knowledge_files = [
        "bio.md",
        "cv.md",
        "faq.md",
        "learning.md",
        "opinions.md",
        "projects.md"
    ]

    for filename in knowledge_files:
        file_path = BASE_DIR / filename
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            documents.append((filename, text))

    if not documents:
        raise ValueError("No Markdown knowledge documents found.")

    return documents

def create_chunks(documents):
    chunks = []

    for filename, text in documents:
        words = text.split()
        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "source": filename,
                "chunk_id": chunk_id,
                "text": chunk_text
            })

            chunk_id += 1
            start += CHUNK_SIZE - OVERLAP

    return chunks


def get_corpus_hash(chunks):
    content = json.dumps(
        chunks,
        ensure_ascii=False,
        sort_keys=True
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_or_create_embeddings(chunks, model):
    corpus_hash = get_corpus_hash(chunks)

    if EMBEDDINGS_FILE.exists() and META_FILE.exists():
        try:
            metadata = json.loads(
                META_FILE.read_text(encoding="utf-8")
            )

            if metadata.get("corpus_hash") == corpus_hash:
                embeddings = np.load(EMBEDDINGS_FILE)

                if len(embeddings) == len(chunks):
                    print("Existing embeddings loaded.")
                    return embeddings
        except (ValueError, OSError, json.JSONDecodeError):
            pass

    print("Creating embeddings for knowledge base...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(EMBEDDINGS_FILE, embeddings)

    META_FILE.write_text(
        json.dumps({"corpus_hash": corpus_hash}, indent=2),
        encoding="utf-8"
    )

    print("Embeddings saved successfully.")

    return embeddings


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


class HybridRAG:
    def __init__(self):
        print("Loading knowledge documents...")

        documents = load_documents()
        self.chunks = create_chunks(documents)

        print(f"Total chunks: {len(self.chunks)}")

        self.model = SentenceTransformer(MODEL_NAME)

        # BM25 index
        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        # Dense embeddings
        self.embeddings = load_or_create_embeddings(
            self.chunks,
            self.model
        )

        print("BM25 and Dense indexes are ready!")

    def retrieve(self, query, mode="Hybrid", top_k=4, alpha=0.5):
        query_tokens = tokenize(query)

        # BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_normalized = normalize_scores(bm25_scores)

        # Dense scores (cosine similarity)
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        dense_scores = self.embeddings @ query_embedding
        dense_normalized = normalize_scores(dense_scores)

        # Select retrieval method
        if mode == "BM25":
            final_scores = bm25_normalized

        elif mode == "Dense":
            final_scores = dense_normalized

        elif mode == "Hybrid":
            final_scores = (
                alpha * dense_normalized
                + (1 - alpha) * bm25_normalized
            )

        else:
            raise ValueError("Mode must be BM25, Dense, or Hybrid.")

        # Highest scoring chunks first
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[index]

            results.append({
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(final_scores[index]),
                "bm25_score": float(bm25_normalized[index]),
                "dense_score": float(dense_normalized[index])
            })

        return results


if __name__ == "__main__":
    rag = HybridRAG()

    question = input("\nAsk a question about yourself: ")

    for mode in ["BM25", "Dense", "Hybrid"]:
        print(f"\n--- {mode} Results ---")

        results = rag.retrieve(
            question,
            mode=mode,
            top_k=3
        )

        for result in results:
            print(
                f"\nSource: {result['source']} | "
                f"Chunk: {result['chunk_id']} | "
                f"Score: {result['score']:.4f}"
            )
            print(result["text"][:500])