"""
Naive RAG vs Advanced RAG(Cohere Rerank) 비교 데모 서버.

실행:
  set SUPABASE_URL=...
  set SUPABASE_KEY=...
  set OPENROUTER_API_KEY=...
  set COHERE_API_KEY=...
  python scripts/rag_compare_server.py --port 5000

브라우저에서 http://127.0.0.1:5000 접속.
"""

import argparse
import concurrent.futures
import os
import sys
import traceback

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_lib  # noqa: E402

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "rag_compare.html")


@app.route("/api/compare", methods=["POST"])
def compare():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question이 비어 있습니다."}), 400

    top_k = int(body.get("top_k", 3))
    candidate_k = int(body.get("candidate_k", 10))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            naive_future = pool.submit(rag_lib.naive_rag, question, top_k)
            advanced_future = pool.submit(rag_lib.advanced_rag, question, candidate_k, top_k)
            naive_result = naive_future.result()
            advanced_result = advanced_future.result()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify({"naive": naive_result, "advanced": advanced_result})


@app.route("/api/health")
def health():
    missing = [
        name
        for name in ["SUPABASE_URL", "SUPABASE_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"]
        if not os.environ.get(name)
    ]
    return jsonify({"ok": not missing, "missing_env": missing})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    missing = [
        name
        for name in ["SUPABASE_URL", "SUPABASE_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"]
        if not os.environ.get(name)
    ]
    if missing:
        print(f"경고: 환경변수 미설정 - {missing} (요청 시 오류가 발생합니다)")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
