"""
data/chunks_page_01_10/all_embedding_chunks.jsonl (Child + Overview) ->
OpenAI text-embedding-3-small 임베딩 생성 -> Supabase documents_test 테이블 적재.
같은 디렉터리의 parents.jsonl(Parent)이 있으면 임베딩 없이 함께 적재한다
(Parent는 벡터 검색 대상이 아니라 chunk_id로 직접 조회되는 "small-to-big" 확장용).

환경변수로 자격증명을 받는다 (하드코딩 금지):
  SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def env_or_exit(name):
    val = os.environ.get(name)
    if not val:
        print(f"오류: 환경변수 {name} 가 설정되지 않았습니다.", file=sys.stderr)
        sys.exit(2)
    return val


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_row(record):
    content = record["embedding_text"]
    metadata = {
        "chunk_id": record["id"],
        "parent_id": record.get("parent_id"),
        "document_id": record.get("document_id"),
        "document_title": record.get("document_title"),
        "document_version": record.get("document_version"),
        "source_file": record.get("source_file"),
        "task_no": record.get("task_no"),
        "field": record.get("field"),
        "core_business": record.get("core_business"),
        "project_name": record.get("project_name"),
        "catalog_project_name": record.get("catalog_project_name"),
        "related_business": record.get("related_business"),
        "departments": record.get("departments"),
        "chunk_type": record.get("chunk_type"),
        "section_label": record.get("section_label"),
        "part_index": record.get("part_index"),
        "part_count": record.get("part_count"),
        "analysis_methods": record.get("analysis_methods"),
        "data_sources": record.get("data_sources"),
        "page_range": record.get("page_range"),
        "source_page_start": record.get("source_page_start"),
        "source_page_end": record.get("source_page_end"),
        "source_start_line": record.get("source_start_line"),
        "source_end_line": record.get("source_end_line"),
        "token_count": record.get("token_count"),
        "embedding_token_count": record.get("embedding_token_count"),
        "content_plain": record.get("content"),
    }
    return content, metadata


def build_parent_row(record):
    """Parent는 벡터 검색 대상이 아니므로 embedding_text 없이 content를 그대로 쓴다."""
    content = record["content"]
    metadata = {
        "chunk_id": record["id"],
        "parent_id": record.get("parent_id"),
        "document_id": record.get("document_id"),
        "document_title": record.get("document_title"),
        "document_version": record.get("document_version"),
        "source_file": record.get("source_file"),
        "task_no": record.get("task_no"),
        "field": record.get("field"),
        "core_business": record.get("core_business"),
        "project_name": record.get("project_name"),
        "catalog_project_name": record.get("catalog_project_name"),
        "related_business": record.get("related_business"),
        "departments": record.get("departments"),
        "chunk_type": record.get("chunk_type", "parent"),
        "section_label": record.get("section_label"),
        "page_range": record.get("page_range"),
        "source_page_start": record.get("source_page_start"),
        "source_page_end": record.get("source_page_end"),
        "source_start_line": record.get("source_start_line"),
        "source_end_line": record.get("source_end_line"),
        "token_count": record.get("token_count"),
        "content_plain": record.get("content"),
    }
    return content, metadata


def call_openai_embeddings(texts, api_key, model, api_base, batch_size=20):
    embeddings = [None] * len(texts)
    endpoint = api_base
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            print(f"OpenAI API 오류 ({e.code}): {err_body}", file=sys.stderr)
            sys.exit(1)
        for item in body["data"]:
            embeddings[start + item["index"]] = item["embedding"]
        print(f"임베딩 진행: {min(start + batch_size, len(texts))}/{len(texts)}")
    return embeddings


def insert_supabase(rows, supabase_url, supabase_key, table):
    endpoint = f"{supabase_url}/rest/v1/{table}"
    payload = json.dumps(rows).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"Supabase 삽입 오류 ({e.code}): {err_body}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="청크 -> 임베딩 -> Supabase 적재")
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default="data/chunks_page_01_10/all_embedding_chunks.jsonl",
    )
    parser.add_argument("--table", default="documents_test")
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument(
        "--api-base",
        default="https://api.openai.com/v1/embeddings",
        help="임베딩 API 엔드포인트. OpenRouter 사용 시 https://openrouter.ai/api/v1/embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--parents-path",
        default=None,
        help="Parent 청크 jsonl 경로 (기본: jsonl_path와 같은 디렉터리의 parents.jsonl, 있으면 자동 적재)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="임베딩/업로드 없이 전처리 결과만 출력",
    )
    args = parser.parse_args()

    records = load_records(args.jsonl_path)
    print(f"입력 레코드: {len(records)}개 ({args.jsonl_path})")

    contents = []
    metadatas = []
    for r in records:
        content, metadata = build_row(r)
        contents.append(content)
        metadatas.append(metadata)

    parents_path = args.parents_path
    if parents_path is None:
        candidate = os.path.join(os.path.dirname(args.jsonl_path), "parents.jsonl")
        parents_path = candidate if os.path.exists(candidate) else None

    parent_records = load_records(parents_path) if parents_path else []
    if parents_path:
        print(f"Parent 레코드: {len(parent_records)}개 ({parents_path})")

    if args.dry_run:
        print(json.dumps({"content": contents[0], "metadata": metadatas[0]}, ensure_ascii=False, indent=2))
        print(f"(dry-run) 총 {len(records)}개 행이 준비되었습니다. 실제 업로드는 수행하지 않았습니다.")
        if parent_records:
            p_content, p_metadata = build_parent_row(parent_records[0])
            print(json.dumps({"content": p_content, "metadata": p_metadata}, ensure_ascii=False, indent=2))
            print(f"(dry-run) Parent {len(parent_records)}개 행이 준비되었습니다 (임베딩 없이 적재 예정).")
        return

    openai_key = env_or_exit("OPENAI_API_KEY")
    supabase_url = env_or_exit("SUPABASE_URL").rstrip("/")
    supabase_key = env_or_exit("SUPABASE_KEY")

    embeddings = call_openai_embeddings(contents, openai_key, args.model, args.api_base, args.batch_size)

    rows = []
    for content, metadata, embedding in zip(contents, metadatas, embeddings):
        rows.append({"content": content, "metadata": metadata, "embedding": embedding})

    result = insert_supabase(rows, supabase_url, supabase_key, args.table)
    print(f"Supabase '{args.table}' 테이블에 {len(result)}개 행 적재 완료.")

    if parent_records:
        parent_rows = []
        for r in parent_records:
            content, metadata = build_parent_row(r)
            parent_rows.append({"content": content, "metadata": metadata, "embedding": None})
        parent_result = insert_supabase(parent_rows, supabase_url, supabase_key, args.table)
        print(f"Supabase '{args.table}' 테이블에 Parent {len(parent_result)}개 행 적재 완료 (임베딩 없음).")


if __name__ == "__main__":
    main()
