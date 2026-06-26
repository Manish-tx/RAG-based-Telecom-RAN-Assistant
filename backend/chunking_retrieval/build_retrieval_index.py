import json
import pickle
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
INDEX_DIR = BASE_DIR / "retrieval_index"
INDEX_DIR.mkdir(exist_ok=True)

# Upgraded to a much stronger embedding model for technical documentation
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Optimized for semantic context while fitting within model token limits
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_json(path: Path) -> Any:
    if not path.exists():
        print(f"Warning: Data file not found at {path}")
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def clean_text(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, (int, float, bool)):
        return str(value)
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_text_from_teleqna(record: Dict[str, Any], idx: int) -> str:
    # Point 3: Removed distractor options, focusing purely on Q&A and explanation
    question = clean_text(record.get("question", ""))
    answer = clean_text(record.get("answer", ""))
    explanation = clean_text(record.get("explanation", ""))
    
    return (
        f"TeleQnA Scenario:\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Explanation: {explanation}"
    )


def build_text_from_3gpp(item: Dict[str, Any]) -> str:
    section_id = clean_text(item.get("section_id", ""))
    title = clean_text(item.get("section_title", ""))
    body = clean_text(item.get("raw_text", ""))
    document_id = clean_text(item.get("document_id", ""))
    release = clean_text(item.get("release_version", ""))
    
    return f"3GPP Document {document_id} Release {release}, Section {section_id} - {title}:\n{body}"


def build_text_from_oran(item: Dict[str, Any]) -> str:
    # Point 1: Converted raw telemetry key-values into rich semantic sentences
    protocol = clean_text(item.get("protocol_stack", "Unknown"))
    throughput = clean_text(item.get("avg_throughput_mbps", "N/A"))
    loss = clean_text(item.get("avg_packet_loss_percent", "N/A"))
    rsrp = clean_text(item.get("avg_rsrp", "N/A"))
    rrc_setup = clean_text(item.get("rrc_setup", "N/A"))
    
    return (
        f"O-RAN Network Summary:\n"
        f"Protocol used was {protocol}.\n"
        f"Average throughput recorded at {throughput} Mbps.\n"
        f"Packet loss experienced: {loss}%.\n"
        f"Average RSRP: {rsrp}.\n"
        f"RRC Setup status: {rrc_setup}.\n"
        f"Observation: This sample represents a network session with packet loss of {loss}% "
        f"and throughput of {throughput} Mbps."
    )


def build_text_from_tust(item: Dict[str, Any]) -> str:
    # Point 2: Converted simulation stats into a semantic narrative
    run_id = clean_text(item.get("run_id", "Unknown"))
    dl_prb = clean_text(item.get("peak_dl_prb_usage_cell", "N/A"))
    avg_delay = clean_text(item.get("avg_dl_delay_ms", "N/A"))
    qos_events = clean_text(item.get("poor_qos_event_count", "N/A"))
    
    return (
        f"Simulation Summary for Run ID: {run_id}.\n"
        f"The network experienced {dl_prb} peak downlink PRB usage.\n"
        f"Average downlink delay was {avg_delay} ms.\n"
        f"There were {qos_events} poor QoS events recorded during this session."
    )


def chunk_documents(texts: List[str], metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    seen_hashes = set()
    global_counter = 0  # Global counter to absolutely guarantee uniqueness
    
    print("\n--- Starting Document Chunking & Deduplication ---")
    for text, metadata in tqdm(zip(texts, metadata_list), total=len(texts), desc="Chunking"):
        split_texts = splitter.split_text(text)
        
        for chunk_idx, chunk_str in enumerate(split_texts):
            chunk_hash = hashlib.md5(chunk_str.encode('utf-8')).hexdigest()
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)
            
            source = metadata.get("source", "unknown")
            if source.startswith("3gpp"):
                doc_id = metadata.get("document", "unknown").replace(" ", "")
                sec_id = metadata.get("section", "unknown").replace(" ", "")
                # Added global_counter to prevent overlapping section numbers colliding
                base_id = f"{source}_{doc_id}_{sec_id}_{global_counter}"
            elif source == "teleqna":
                base_id = f"teleqna_{metadata.get('id', 'unknown')}"
            elif source == "oran":
                base_id = f"oran_{metadata.get('id', 'unknown')}"
            elif source == "tust":
                base_id = f"tust_{metadata.get('id', 'unknown')}"
            else:
                base_id = f"gen_{chunk_hash[:8]}"
            
            chunk_id = f"{base_id}_{chunk_idx:04d}"
            chunk_meta = {k: str(v) for k, v in metadata.items() if v is not None}
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_str,
                "metadata": chunk_meta
            })
            
            global_counter += 1  # Increment for every unique text chunk saved
            
    return chunks

def create_documents() -> List[Dict[str, Any]]:
    document_texts: List[str] = []
    document_metadata: List[Dict[str, Any]] = []

    # 3GPP Release 16
    print("Loading 3GPP Rel 16...")
    rel16 = load_json(DATA_DIR / "parsed_3gpp_data_version16.json")
    for item in rel16:
        document_texts.append(build_text_from_3gpp(item))
        # Point 5: Enriched metadata schema
        document_metadata.append({
            "id": f"rel16_{item.get('section_id', 'unknown')}",
            "source": "3gpp",
            "release": "16",
            "section": item.get("section_id", "unknown"),
            "title": item.get("section_title", "unknown"),
            "document": item.get("document_id", "unknown")
        })

    # 3GPP Release 18
    print("Loading 3GPP Rel 18...")
    rel18 = load_json(DATA_DIR / "parsed_38331_Rel18.json")
    for item in rel18:
        document_texts.append(build_text_from_3gpp(item))
        document_metadata.append({
            "id": f"rel18_{item.get('section_id', 'unknown')}",
            "source": "3gpp",
            "release": "18",
            "section": item.get("section_id", "unknown"),
            "title": item.get("section_title", "unknown"),
            "document": item.get("document_id", "unknown")
        })

    # TeleQnA
    print("Loading TeleQnA...")
    teleqna = load_json(DATA_DIR / "teleqna.json")
    if isinstance(teleqna, dict):
        for key, item in teleqna.items():
            document_texts.append(build_text_from_teleqna(item, 0))
            document_metadata.append({
                "id": key,
                "source": "teleqna",
                "category": item.get("category", "unknown"),
            })

    # O-RAN anomaly logs
    print("Loading O-RAN Logs...")
    oran = load_json(DATA_DIR / "o-ran_anomaly.json")
    for idx, item in enumerate(oran):
        document_texts.append(build_text_from_oran(item))
        document_metadata.append({
            "id": item.get("log_id", str(idx)),
            "source": "oran",
            "protocol_stack": item.get("protocol_stack", "unknown"),
            "flow_type": item.get("flow_type", "unknown"),
        })

    # Small TuST summaries
    print("Loading TuST Summaries...")
    tust = load_json(DATA_DIR / "small_tust_processed.json")
    for idx, item in enumerate(tust):
        document_texts.append(build_text_from_tust(item))
        document_metadata.append({
            "id": item.get("run_id", str(idx)),
            "source": "tust",
            "anomaly_flag": str(item.get("anomaly_flag", "unknown")),
        })

    return [{"text": text, "metadata": meta} for text, meta in zip(document_texts, document_metadata)]


def main() -> None:
    docs = create_documents()
    chunks = chunk_documents([d["text"] for d in docs], [d["metadata"] for d in docs])

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]

    print(f"\nCreated {len(texts)} unique, deduplicated chunks.")
    import torch 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- Generating Embeddings using: {device.upper()} ({EMBEDDING_MODEL}) ---")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    # Added tqdm to embedding creation, handling it in batches if necessary, 
    # but encode handles progress tracking if show_progress_bar is True
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.tolist()

    print("\n--- Saving to ChromaDB ---")
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection("telecom_rag")
    except Exception:
        pass
    
    # Point 9: Add HNSW cosine metadata
    collection = client.create_collection(
        name="telecom_rag",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Batch adding to chroma DB with tqdm just to be safe with large datasets
    batch_size = 5000
    for i in tqdm(range(0, len(texts), batch_size), desc="Writing to ChromaDB"):
        collection.add(
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size]  # Using meaningful IDs
        )

    print("\n--- Generating BM25 Index ---")
    tokenized_docs = []
    for text in tqdm(texts, desc="Tokenizing for BM25"):
        tokenized_docs.append(re.findall(r"\w+", text.lower()))
        
    bm25 = BM25Okapi(tokenized_docs)
    with (INDEX_DIR / "bm25_index.pkl").open("wb") as fh:
        pickle.dump({"texts": texts, "tokenized_docs": tokenized_docs, "bm25": bm25, "ids": ids}, fh)

    print(f"\nSuccessfully saved Hybrid RAG indices to {INDEX_DIR}")


if __name__ == "__main__":
    main()