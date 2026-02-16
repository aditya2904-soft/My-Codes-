# fmp_tfidf_semantic_app.py
import os
import hashlib
import pickle
from typing import List, Dict
from datetime import datetime

import streamlit as st
import requests
import pandas as pd
import numpy as np
import textwrap

# TF-IDF / sparse utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

# -----------------------
# Configuration
# -----------------------
st.set_page_config(page_title="FMP Transcript + TF-IDF Search", layout="wide")

API_URL_TEMPLATE = (
    "https://financialmodelingprep.com/stable/earning-call-transcript"
    "?symbol={symbol}&year={year}&quarter={quarter}&apikey={apikey}"
)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
INDEX_DIR = os.environ.get("INDEX_DIR", "fmp_tfidf_index")
os.makedirs(INDEX_DIR, exist_ok=True)

# TF-IDF artifact paths
TFIDF_VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.npz")
TFIDF_META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

# -----------------------
# Utility helpers
# -----------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Dict]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunk_id = f"{sha1(chunk)}_{start}"
        chunks.append({"chunk_id": chunk_id, "text": chunk, "start": start, "end": end})
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

# -----------------------
# FMP fetcher (original)
# -----------------------
@st.cache_data(ttl=60 * 60)
def fetch_transcript(symbol: str, year: int, quarter: int, apikey: str):
    if not apikey:
        raise ValueError("No API key provided. Please provide an API key in the sidebar or set FMP_API_KEY.")
    url = API_URL_TEMPLATE.format(symbol=symbol, year=year, quarter=quarter, apikey=apikey)
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception as e:
            raise ValueError(f"Failed to decode JSON response: {e}")
        return data
    elif resp.status_code == 401:
        raise PermissionError("Unauthorized: Invalid API key or access denied.")
    else:
        raise ConnectionError(f"API request failed (status {resp.status_code}): {resp.text}")

def extract_primary_items(json_payload):
    if isinstance(json_payload, list):
        rows = []
        for item in json_payload:
            rows.append({
                "symbol": item.get("symbol"),
                "date": item.get("date"),
                "title": item.get("title") or item.get("transcriptTitle") or None,
                "type": item.get("type"),
                "transcript": item.get("transcript") or item.get("content") or item.get("text"),
                "raw": item,
            })
        return pd.DataFrame(rows)
    if isinstance(json_payload, dict):
        item = json_payload
        row = {
            "symbol": item.get("symbol"),
            "date": item.get("date"),
            "title": item.get("title") or item.get("transcriptTitle") or None,
            "type": item.get("type"),
            "transcript": item.get("transcript") or item.get("content") or item.get("text"),
            "raw": item,
        }
        return pd.DataFrame([row])
    return pd.DataFrame([])

# -----------------------
# TF-IDF index functions
# -----------------------
def build_or_load_index_tfidf():
    vectorizer = None
    matrix = None
    meta = {"metadatas": []}
    if os.path.exists(TFIDF_VECTORIZER_PATH) and os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(TFIDF_META_PATH):
        try:
            with open(TFIDF_VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
            matrix = sp.load_npz(TFIDF_MATRIX_PATH)
            with open(TFIDF_META_PATH, "rb") as f:
                meta = pickle.load(f)
        except Exception:
            vectorizer, matrix, meta = None, None, {"metadatas": []}
    return vectorizer, matrix, meta

def save_tfidf_index(vectorizer, matrix, meta):
    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(TFIDF_MATRIX_PATH, matrix)
    with open(TFIDF_META_PATH, "wb") as f:
        pickle.dump(meta, f)

def index_transcript_tfidf(vectorizer, matrix, meta, transcript_text: str, metadata: dict):
    chunks = chunk_text(transcript_text)
    if not chunks:
        return vectorizer, matrix, meta, 0

    texts = [c["text"] for c in chunks]

    if vectorizer is None:
        # Fit initial vectorizer on these chunks
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000, stop_words="english")
        X_new = vectorizer.fit_transform(texts)
        matrix = X_new
    else:
        # Transform new chunks and append
        X_new = vectorizer.transform(texts)
        if matrix is None:
            matrix = X_new
        else:
            matrix = sp.vstack([matrix, X_new])

    # normalize for cosine similarity
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)

    # append metadata records aligned with matrix rows
    for c in chunks:
        md = {**metadata, **c}
        meta["metadatas"].append(md)

    return vectorizer, matrix, meta, len(chunks)

def query_index_tfidf(vectorizer, matrix, meta, query: str, top_k: int = 5):
    if vectorizer is None or matrix is None or matrix.shape[0] == 0:
        return []
    q_vec = vectorizer.transform([query])
    q_vec = normalize(q_vec, norm="l2", axis=1, copy=False)
    scores = (matrix @ q_vec.T).toarray().ravel()
    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for idx_pos in top_idx:
        score = float(scores[idx_pos])
        md = meta["metadatas"][idx_pos]
        results.append({
            "score": score,
            "chunk_id": md.get("chunk_id"),
            "text": md.get("text"),
            "start": md.get("start"),
            "end": md.get("end"),
            **{k: md.get(k) for k in ("symbol", "date", "title", "year", "quarter")}
        })
    return results

# -----------------------
# Streamlit layout
# -----------------------
st.title("Earnings Call — Transcript + TF-IDF Search")
st.markdown("Fetch transcript from FinancialModelingPrep, view the *total transcript*, index with TF-IDF, and run searches.")

# Sidebar: fetcher settings
with st.sidebar.form("fetch_form"):
    ticker = st.text_input("Ticker symbol (e.g. MNDY)", value="MNDY").strip().upper()
    year = st.number_input("Year", min_value=1990, max_value=2099, value=2024, step=1)
    quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=1)
    default_key = os.environ.get("FMP_API_KEY", "")
    api_key = st.text_input("FMP API Key", value=default_key, type="password")
    fetch_btn = st.form_submit_button("Fetch Transcript")
st.sidebar.markdown("---")
st.sidebar.markdown("Index settings (TF-IDF):")
st.sidebar.write(f"chunk size: {CHUNK_SIZE} • overlap: {CHUNK_OVERLAP}")
st.sidebar.markdown("---")
st.sidebar.caption("Tip: set FMP_API_KEY env var to avoid pasting each session.")

# load or init TF-IDF index
vectorizer, matrix, meta = build_or_load_index_tfidf()

# Sample transcript for local testing
SAMPLE_TRANSCRIPT = (
    "Operator: Welcome to the quarterly call. All lines are muted.\n\n"
    "CEO: Good morning everyone. Thank you for joining us. Our Q2 results were in line with expectations..."
    "\nCFO: We saw revenue growth driven by product X and international expansion."
    "\nAnalyst: Can you comment on churn?\nCEO: Churn improved due to product enhancements..."
)

# Fetch transcripts
if fetch_btn:
    try:
        payload = fetch_transcript(ticker, int(year), int(quarter), api_key)
        if not payload:
            st.warning("No transcript returned for the selected ticker/year/quarter.")
            df = pd.DataFrame([])
        else:
            df = extract_primary_items(payload)
            st.success(f"Fetched {len(df)} item(s) for {ticker} — Year {year} Q{quarter}")
    except Exception as e:
        st.error(f"Fetch error: {e}")
        df = pd.DataFrame([])
else:
    df = pd.DataFrame([])

# If fetched, display metadata and let user choose one to view + index
if not df.empty:
    st.markdown("### Fetched transcripts (metadata)")
    meta_cols = [c for c in df.columns if c != "raw"]
    st.dataframe(df[meta_cols].fillna("-"))

    idx_choice = st.number_input("Select row index to view transcript", min_value=0, max_value=max(0, len(df) - 1), value=0, step=1)
    selected = df.iloc[int(idx_choice)]
    st.subheader(f"Full transcript — {selected.get('title') or ticker}")
    transcript_text = selected.get("transcript") or ""
    if not transcript_text:
        st.info("No transcript text found; showing raw JSON below.")
        with st.expander("Raw JSON response"):
            st.json(selected.get("raw"))
    else:
        st.download_button(label="Download full transcript (.txt)",
                           data=transcript_text,
                           file_name=f"{ticker}_{year}_Q{quarter}_transcript.txt",
                           mime="text/plain")
        st.text_area("Total transcript (read-only)", value=transcript_text, height=450)

        index_now = st.button("Index this transcript (TF-IDF)")
        if index_now:
            tx_hash = sha1(transcript_text)[:12]
            existing_hashes = {m.get("_tx_hash") for m in meta.get("metadatas", []) if m.get("_tx_hash")}
            if tx_hash in existing_hashes:
                st.info("This transcript already indexed (hash match). Skipping.")
            else:
                metadata = {
                    "symbol": ticker,
                    "date": selected.get("date"),
                    "title": selected.get("title"),
                    "year": year,
                    "quarter": quarter,
                    "_tx_hash": tx_hash
                }
                vectorizer, matrix, meta, added = index_transcript_tfidf(vectorizer, matrix, meta, transcript_text, metadata)
                save_tfidf_index(vectorizer, matrix, meta)
                st.success(f"Indexed {added} chunks (hash={tx_hash}).")
else:
    st.markdown("### No fetched transcript loaded")
    st.info("Paste a transcript below for local testing or use the canned sample.")
    use_sample = st.checkbox("Use canned sample transcript for demo", value=False)
    if use_sample:
        transcript_text = SAMPLE_TRANSCRIPT
    else:
        transcript_text = st.text_area("Paste transcript text here (for indexing/testing)", height=300)
    if transcript_text and st.button("Index pasted transcript"):
        metadata = {"symbol": "SAMPLE", "date": datetime.utcnow().isoformat(), "title": "SAMPLE_TRANSCRIPT", "year": 2024, "quarter": 1, "_tx_hash": sha1(transcript_text)[:12]}
        vectorizer, matrix, meta, added = index_transcript_tfidf(vectorizer, matrix, meta, transcript_text, metadata)
        save_tfidf_index(vectorizer, matrix, meta)
        st.success(f"Indexed {added} chunks from pasted transcript.")

st.markdown("---")
# Search UI
st.subheader("TF-IDF Search (ask a question)")
query = st.text_input("Enter your question / search phrase", value="", key="semantic_query")
top_k = st.slider("Top K results", min_value=1, max_value=10, value=5)
search_btn = st.button("Run search")

if search_btn:
    if not query.strip():
        st.warning("Enter a query.")
    else:
        if vectorizer is None or matrix is None or matrix.shape[0] == 0:
            st.info("No transcripts indexed yet. Index via fetcher or paste a transcript.")
        else:
            results = query_index_tfidf(vectorizer, matrix, meta, query, top_k=top_k)
            if not results:
                st.info("No matches found.")
            else:
                for i, r in enumerate(results):
                    st.markdown(f"**Result {i+1} — score: {r['score']:.4f} — {r.get('symbol','')} {r.get('date','')}**")
                    snippet = textwrap.shorten(r["text"], width=300, placeholder="...")
                    st.write(snippet)
                    with st.expander("Show full chunk + metadata"):
                        st.write(r["text"])
                        st.json(r)

                st.markdown("---")
                st.info("Optional: synthesize an answer from retrieved passages via OpenAI (API key required).")

                openai_key = st.text_input("OpenAI API key (optional)", type="password")
                if openai_key:
                    try:
                        import importlib
                        try:
                            openai = importlib.import_module("openai")
                        except ModuleNotFoundError:
                            st.error("The 'openai' package is not installed. Install with `pip install openai` and restart.")
                            openai = None
                        if openai is not None:
                            openai.api_key = openai_key
                            context = "\n\n---\n\n".join([f"[{i+1}] chunk_id:{r['chunk_id']}\n{r['text']}" for i,r in enumerate(results)])
                            prompt = (
                                "You are a precise research assistant. Use only the provided context; do not invent facts. "
                                "Answer the user's question concisely and list chunk_ids used as citations.\n\n"
                                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                            )
                            completion = openai.ChatCompletion.create(
                                model="gpt-4o-mini",  # change to model you have access to if needed
                                messages=[{"role":"user","content":prompt}],
                                temperature=0.0,
                                max_tokens=400
                            )
                            answer = completion["choices"][0]["message"]["content"].strip()
                            st.markdown("### Synthesized answer (LLM)")
                            st.write(answer)
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")

# persist index state before exit / periodically
if vectorizer is not None and matrix is not None:
    try:
        save_tfidf_index(vectorizer, matrix, meta)
    except Exception:
        pass

st.markdown("---")
st.caption("Built for local testing with FinancialModelingPrep's earning-call-transcript endpoint. Developed by Aditya Shivhare.")
