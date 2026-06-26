import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX_DIR = Path(__file__).resolve().parent / "retrieval_index"

class TelecomRetriever:
    def __init__(self, rrf_k: int = 60, rerank_pool_size: int = 15):
        self.rrf_k = rrf_k
        self.rerank_pool_size = rerank_pool_size
        
        print("Initializing Telecom Hybrid Retriever...")
        
        # 1. Initialize Vector Database Client (FIXED)
        self.chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.collection = self.chroma_client.get_collection(name="telecom_rag")
        
        # 2. Load Sparse BM25 Index
        print("Loading BM25 Keyword Index...")
        with (INDEX_DIR / "bm25_index.pkl").open("rb") as fh:
            data = pickle.load(fh)
            self.texts = data["texts"]
            self.ids = data["ids"]
            self.bm25 = data["bm25"]
            
        # 3. Load Models onto GPU for Real-Time Execution
        print("Loading AI Models onto GPU...")
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        
        print("Retriever Engine Ready.")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, top_k: int = 25, final_k: int = 5) -> List[Dict[str, Any]]:
        # ---------------------------------------------------------
        # STEP 1: DENSE SEMANTIC SEARCH (Concept Matching)
        # ---------------------------------------------------------
        query_vector = self.embedding_model.encode(query, convert_to_tensor=False).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_vector], 
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        # ---------------------------------------------------------
        # STEP 2: SPARSE KEYWORD SEARCH (Exact Acronym Matching)
        # ---------------------------------------------------------
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get indices of the highest scoring BM25 hits
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

        # ---------------------------------------------------------
        # STEP 3: RECIPROCAL RANK FUSION (RRF)
        # ---------------------------------------------------------
        candidate_map = {}
        
        # 3A. Add Dense Ranks to Fusion Math
        if vector_results["ids"] and vector_results["ids"][0]:
            dense_ids = vector_results["ids"][0]
            dense_docs = vector_results["documents"][0]
            dense_metas = vector_results["metadatas"][0]
            
            for rank, (doc_id, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
                if doc_id not in candidate_map:
                    candidate_map[doc_id] = {"id": doc_id, "text": doc, "metadata": meta, "rrf_score": 0.0}
                candidate_map[doc_id]["rrf_score"] += 1.0 / (self.rrf_k + (rank + 1))
                
        # 3B. Add Sparse Ranks to Fusion Math
        for rank, idx in enumerate(top_bm25_indices):
            doc_id = self.ids[idx]
            if doc_id not in candidate_map:
                candidate_map[doc_id] = {"id": doc_id, "text": self.texts[idx], "metadata": {}, "rrf_score": 0.0}
            candidate_map[doc_id]["rrf_score"] += 1.0 / (self.rrf_k + (rank + 1))

        # 3C. Sort by the mathematically fused score and slice the pool
        candidates = list(candidate_map.values())
        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        pool_to_rerank = candidates[:self.rerank_pool_size]

        # ---------------------------------------------------------
        # STEP 4: CROSS-ENCODER DEEP RE-RANKING
        # ---------------------------------------------------------
        pairs = [[query, c["text"]] for c in pool_to_rerank]
        rerank_scores = self.reranker.predict(pairs)
        
        # Attach new scores and sort one final time
        for i, score in enumerate(rerank_scores):
            pool_to_rerank[i]["rerank_score"] = float(score)
            
        pool_to_rerank.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return pool_to_rerank[:final_k]

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    retriever = TelecomRetriever()
    
    test_query = "What causes an RRC Connection Setup failure and high packet loss in O-RAN?"
    print(f"\nExecuting Query: '{test_query}'\n")
    
    results = retriever.retrieve(test_query, top_k=25, final_k=3)
    
    for i, res in enumerate(results, 1):
        print(f"--- Rank {i} (Score: {res['rerank_score']:.4f}) ---")
        print(f"ID: {res['id']}")
        print(f"Text: {res['text'][:150]}...\n")