"""
Naive RAG / Advanced RAG(Rerank) 공통 파이프라인.

외부 서비스:
  - Supabase(pgvector documents_test): 벡터 검색
  - OpenRouter(openai/text-embedding-3-small): 임베딩
  - OpenRouter(openai/gpt-5-mini): 답변 생성
  - Cohere(rerank-v3.5): 재순위화 (Advanced RAG 전용)

자격증명은 환경변수로만 받는다: SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY, COHERE_API_KEY
"""

import os
import time
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-5-mini"
RERANK_MODEL = "rerank-v3.5"
TABLE_NAME = "documents_test"
MATCH_FN = "match_documents_test"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
COHERE_BASE = "https://api.cohere.com/v2"


def _require_config():
    missing = [
        name
        for name, val in [
            ("SUPABASE_URL", SUPABASE_URL),
            ("SUPABASE_KEY", SUPABASE_KEY),
            ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
            ("COHERE_API_KEY", COHERE_API_KEY),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(f"환경변수가 설정되지 않았습니다: {', '.join(missing)}")


def embed_query(text):
    resp = requests.post(
        f"{OPENROUTER_BASE}/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def vector_search(embedding, match_count):
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/{MATCH_FN}",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        },
        json={"query_embedding": embedding, "match_count": match_count, "filter": {}},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    results = []
    for r in rows:
        md = r.get("metadata") or {}
        if md.get("chunk_type") == "parent":
            # Parent는 벡터 검색 대상이 아니라 parent_id로 직접 조회되는 확장용 청크.
            continue
        results.append(
            {
                "chunk_id": md.get("chunk_id"),
                "parent_id": md.get("parent_id"),
                "project_name": md.get("project_name"),
                "section_label": md.get("section_label"),
                "chunk_type": md.get("chunk_type"),
                "task_no": md.get("task_no"),
                "content": r.get("content"),
                "content_plain": md.get("content_plain"),
                "similarity": r.get("similarity"),
            }
        )
    return results


def fetch_parent_chunks(parent_ids):
    unique_ids = sorted({pid for pid in parent_ids if pid})
    if not unique_ids:
        return {}
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
        params={
            "select": "content,metadata",
            "metadata->>chunk_id": f"in.({','.join(unique_ids)})",
        },
        timeout=30,
    )
    resp.raise_for_status()
    parents = {}
    for row in resp.json():
        md = row.get("metadata") or {}
        chunk_id = md.get("chunk_id")
        if chunk_id:
            parents[chunk_id] = row.get("content")
    return parents


def expand_with_parents(chunks):
    parent_map = fetch_parent_chunks(c.get("parent_id") for c in chunks)
    for c in chunks:
        c["parent_content"] = parent_map.get(c.get("parent_id"))
    return chunks


def cohere_rerank(query, candidates, top_n):
    documents = [c["content"] for c in candidates]
    resp = requests.post(
        f"{COHERE_BASE}/rerank",
        headers={
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": RERANK_MODEL, "query": query, "top_n": top_n, "documents": documents},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    reranked = []
    for r in results:
        item = dict(candidates[r["index"]])
        item["rerank_score"] = r["relevance_score"]
        reranked.append(item)
    return reranked


def chat_completion(question, context_chunks):
    context_text = "\n\n".join(
        f"[근거 {i+1}] ({c.get('project_name') or '문서 개요'} · {c.get('section_label') or ''})\n"
        f"{c.get('parent_content') or c['content']}"
        for i, c in enumerate(context_chunks)
    )
    system_prompt = (
        "당신은 '2026년 시정 핵심사업 데이터 분석 컨설팅 제안(안)' 문서에 대해 답변하는 어시스턴트입니다. "
        "아래 제공된 근거 자료만을 근거로 한국어로 답변하세요. 근거에 없는 내용은 추측하지 말고 "
        "'제공된 자료에서 확인할 수 없습니다'라고 답하세요. 답변 끝에 참고한 근거 번호를 [근거 N] 형태로 표시하세요."
    )
    user_prompt = f"질문: {question}\n\n[근거 자료]\n{context_text}"

    resp = requests.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    answer = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return answer, usage


def naive_rag(question, top_k=3):
    _require_config()
    timings = {}

    t0 = time.perf_counter()
    embedding = embed_query(question)
    timings["embedding_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    retrieved = vector_search(embedding, top_k)
    timings["vector_search_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    expand_with_parents(retrieved)
    timings["parent_fetch_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    answer, usage = chat_completion(question, retrieved)
    timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    timings["total_ms"] = round(sum(timings.values()), 1)
    return {
        "method": "naive",
        "question": question,
        "retrieved": retrieved,
        "answer": answer,
        "usage": usage,
        "timings": timings,
    }


def advanced_rag(question, candidate_k=10, top_k=3):
    _require_config()
    timings = {}

    t0 = time.perf_counter()
    embedding = embed_query(question)
    timings["embedding_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    candidates = vector_search(embedding, candidate_k)
    timings["vector_search_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    reranked = cohere_rerank(question, candidates, top_k)
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    expand_with_parents(reranked)
    timings["parent_fetch_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    answer, usage = chat_completion(question, reranked)
    timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    timings["total_ms"] = round(sum(timings.values()), 1)
    return {
        "method": "advanced",
        "question": question,
        "candidates": candidates,
        "retrieved": reranked,
        "answer": answer,
        "usage": usage,
        "timings": timings,
    }
