"""
Markdown -> Vector DB Parent/Child 청킹 스크립트

'2026년 시정 핵심사업 데이터 분석 컨설팅 제안(안)' 문서 전용.
문서의 분석과제 단위(Parent)와 소제목 단위(Child) 구조를 우선 이용해 청킹한다.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _ENC = None


# ----------------------------------------------------------------------------
# 토큰 계산
# ----------------------------------------------------------------------------

_HANGUL_RE = re.compile(r"[가-힣]")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    # tiktoken이 없을 때의 한국어 문자 기반 fallback.
    hangul = len(_HANGUL_RE.findall(text))
    other = len(text) - hangul
    return max(1, math.ceil(hangul * 0.7 + other / 3.5))


TOKEN_METHOD = "tiktoken(cl100k_base)" if _ENC is not None else "korean_char_fallback"


# ----------------------------------------------------------------------------
# 공통 전처리 유틸
# ----------------------------------------------------------------------------

BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
BULLET_LEAD_RE = re.compile(r"^\s*(ㅇ|○|•|o(?=\s))\s?")
NOTE_LEAD_RE = re.compile(r"^\s*※\s*")
TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
SEP_CELL_RE = re.compile(r"^:?-+:?$")

SECTION_LABELS = [
    "추진배경 및 목적",
    "주요내용",
    "분석결과(예시)",
    "타기관 분석사례",
    "실국 활용방안(예시)",
]
SECTION_HEADER_RE = re.compile(
    r"^\W?\s*(" + "|".join(re.escape(s) for s in SECTION_LABELS) + r")\s*$"
)

CHUNK_TYPE_LABELS = {
    "policy_context": "정책배경·목적",
    "analysis_design": "분석설계·활용데이터",
    "expected_results": "예상결과·유사사례",
    "administrative_use": "실국 활용방안",
}


def split_cells(row_line: str):
    m = TABLE_ROW_RE.match(row_line.strip())
    if not m:
        return None
    inner = m.group(1)
    return [c.strip() for c in inner.split("|")]


def is_separator_row(cells):
    return all(SEP_CELL_RE.match(c) for c in cells if c != "") and any(
        "-" in c for c in cells
    )


def normalize_line(line: str) -> str:
    return BR_RE.sub("\n", line)


def clean_bullet_or_note(line: str) -> Optional[str]:
    """일반 텍스트 줄을 목록/참고 표기로 정규화. 표 줄은 여기서 다루지 않는다."""
    line = normalize_line(line)
    parts = [p for p in line.split("\n")]
    out_lines = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if NOTE_LEAD_RE.match(stripped):
            rest = NOTE_LEAD_RE.sub("", stripped)
            out_lines.append(f"- 참고: {rest}")
        elif BULLET_LEAD_RE.match(stripped):
            rest = BULLET_LEAD_RE.sub("", stripped)
            out_lines.append(f"- {rest}")
        else:
            out_lines.append(stripped)
    if not out_lines:
        return None
    return "\n".join(out_lines)


def dedupe_cells(cells):
    seen = []
    for c in cells:
        c = normalize_line(c).strip()
        if not c:
            continue
        if c not in seen:
            seen.append(c)
    return seen


# ----------------------------------------------------------------------------
# 표 처리기: 소제목 버킷 내부에서 만나는 표를 유형별로 텍스트로 변환
# ----------------------------------------------------------------------------

def handle_core_goal_table(rows):
    """단일 열짜리 핵심목표 표 -> 문장 하나."""
    values = dedupe_cells([c for row in rows for c in row])
    if not values:
        return None
    return "핵심목표: " + " ".join(values)


def handle_methods_data_table(rows):
    """(빈칸, 분석기술) / (빈칸, 활용데이터) 2행 표."""
    if len(rows) < 2:
        return None, [], []
    methods_raw = rows[0][-1] if rows[0] else ""
    data_raw = rows[1][-1] if rows[1] else ""

    def parse_methods(raw):
        raw = normalize_line(raw)
        items = []
        for segment in raw.split("\n"):
            segment = segment.strip()
            if not segment:
                continue
            segment = re.sub(r"^[•\-]\s*", "", segment)
            for token in segment.split(","):
                token = token.strip()
                token = re.sub(r"\s*등\s*$", "", token)
                if token:
                    items.append(token)
        return items

    def parse_data_sources(raw):
        raw = normalize_line(raw)
        items = []
        for segment in raw.split("\n"):
            segment = segment.strip()
            if not segment:
                continue
            segment = re.sub(r"^[•\-]\s*", "", segment)
            if segment:
                items.append(segment)
        return items

    methods = parse_methods(methods_raw)
    data_sources = parse_data_sources(data_raw)

    text_lines = []
    if methods:
        text_lines.append("분석기술:")
        text_lines.extend(f"- {m}" for m in methods)
    if data_sources:
        text_lines.append("활용데이터:")
        text_lines.extend(f"- {d}" for d in data_sources)
    return ("\n".join(text_lines) if text_lines else None), methods, data_sources


def handle_viz_title_table(rows):
    """시각화 제목 표 -> 시각화 예시 목록. 이미지 행은 버린다."""
    header = rows[0] if rows else []
    titles = dedupe_cells(header)
    titles = [t for t in titles if not IMAGE_RE.search(t)]
    if not titles:
        return None
    lines = ["시각화 예시:"]
    lines.extend(f"- {t}" for t in titles)
    return "\n".join(lines)


def handle_scenario_table(rows):
    """활용 시나리오 표: 마지막 데이터 행의 중복 셀을 하나로 합친다."""
    if not rows:
        return None
    last_row = rows[-1]
    values = dedupe_cells(last_row)
    values = [v for v in values if "활용 시나리오" not in v]
    if not values:
        return None
    return "활용 시나리오:\n" + "\n".join(values)


def handle_generic_table(rows):
    lines = []
    for row in rows:
        values = dedupe_cells(row)
        values = [v for v in values if not IMAGE_RE.search(v)]
        if values:
            lines.append(" / ".join(values))
    return "\n".join(lines) if lines else None


def classify_and_render_table(bucket_name, rows):
    """rows: 구분선을 제외한 셀 리스트의 리스트. 반환: (텍스트 or None, methods, data_sources)"""
    non_empty_flat = [c for row in rows for c in row if c.strip()]
    has_scenario_marker = any("활용 시나리오" in c for c in non_empty_flat)
    if has_scenario_marker:
        return handle_scenario_table(rows), [], []

    if len(rows) == 1 and all(len(r) == 1 for r in rows):
        return handle_core_goal_table(rows), [], []

    if (
        len(rows) == 2
        and all(len(r) >= 2 for r in rows)
        and rows[0][0] == ""
        and rows[1][0] == ""
    ):
        text, methods, data_sources = handle_methods_data_table(rows)
        return text, methods, data_sources

    if len(rows) >= 2:
        header = rows[0]
        data_rows = rows[1:]
        header_is_titles = all(c.strip() and not IMAGE_RE.search(c) for c in header)
        data_looks_like_images = all(
            (not c.strip()) or IMAGE_RE.search(c) for r in data_rows for c in r
        )
        if header_is_titles and data_looks_like_images:
            return handle_viz_title_table(rows), [], []

    return handle_generic_table(rows), [], []


# ----------------------------------------------------------------------------
# 상단 과제 목록(카탈로그) 파싱
# ----------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    task_no: int
    field: str
    core_business: str
    project_name: str
    departments: list = field(default_factory=list)


def parse_catalog(lines, start_idx, end_idx):
    entries = []
    task_no = 0
    for i in range(start_idx, end_idx):
        raw = lines[i]
        cells = split_cells(raw)
        if not cells or len(cells) != 5:
            continue
        if is_separator_row(cells):
            continue
        if cells[0] == "연번":
            continue
        task_no += 1
        dept_raw = normalize_line(cells[4])
        departments = [d.strip() for d in dept_raw.split("\n") if d.strip()]
        entries.append(
            CatalogEntry(
                task_no=task_no,
                field=cells[1].strip(),
                core_business=cells[2].strip(),
                project_name=cells[3].strip(),
                departments=departments,
            )
        )
    return entries


# ----------------------------------------------------------------------------
# 본문 분석과제 블록 파싱
# ----------------------------------------------------------------------------

TITLE_BAR_RE = re.compile(r"^\|\s*분석과제(\d+)\s*\|\s*(.+?)\s*\|\s*$")
RELATED_BIZ_RE = re.compile(r"^\|\s*연관사업\s*\|\s*(.+?)\s*\|\s*$")
DEPT_SUFFIX_RE = re.compile(r"^(.*)\(([^()]+)\)\s*$")


@dataclass
class TaskBlock:
    task_no: int
    project_name: str
    start_line: int  # 0-indexed
    end_line: int  # 0-indexed, exclusive
    body_lines: list
    related_business_raw: str = ""
    related_business: str = ""
    related_dept_text: str = ""


def find_task_blocks(lines):
    """전체 문서에서 '| 분석과제N | ... |' 로 시작하는 모든 블록의 경계를 찾는다."""
    starts = []
    for i, line in enumerate(lines):
        m = TITLE_BAR_RE.match(line.rstrip("\n"))
        if m:
            starts.append((i, int(m.group(1)), m.group(2).strip()))

    blocks = []
    for idx, (line_no, task_no, project_name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        blocks.append(
            TaskBlock(
                task_no=task_no,
                project_name=project_name,
                start_line=line_no,
                end_line=end,
                body_lines=lines[line_no:end],
            )
        )
    return blocks


def extract_related_business(block: TaskBlock):
    for line in block.body_lines[:8]:
        m = RELATED_BIZ_RE.match(line.rstrip("\n"))
        if m:
            raw = m.group(1).strip()
            dm = DEPT_SUFFIX_RE.match(raw)
            if dm:
                block.related_business = dm.group(1).strip()
                block.related_dept_text = dm.group(2).strip()
            else:
                block.related_business = raw
            block.related_business_raw = raw
            return


def segment_body(block: TaskBlock):
    """제목줄/연관사업 표 이후 부분을 소제목 버킷으로 나눈다."""
    buckets = {label: [] for label in SECTION_LABELS}
    order = []

    # 연관사업 표까지 건너뛰기: 첫 '| 분석과제' 줄과 '| 연관사업' 줄 다음부터 처리.
    started = False
    current_bucket = None
    i = 0
    n = len(block.body_lines)
    while i < n:
        raw_line = block.body_lines[i].rstrip("\n")

        if not started:
            if RELATED_BIZ_RE.match(raw_line):
                started = True
            i += 1
            continue

        stripped = raw_line.strip()
        header_m = SECTION_HEADER_RE.match(stripped)
        if header_m:
            current_bucket = header_m.group(1)
            if current_bucket not in order:
                order.append(current_bucket)
            i += 1
            continue

        if current_bucket is None:
            i += 1
            continue

        if stripped == "":
            i += 1
            continue

        if TABLE_ROW_RE.match(stripped):
            table_lines = []
            while i < n:
                s2 = block.body_lines[i].rstrip("\n").strip()
                if s2 == "" or not TABLE_ROW_RE.match(s2):
                    break
                table_lines.append(s2)
                i += 1
            rows = []
            for tl in table_lines:
                cells = split_cells(tl)
                if cells is None:
                    continue
                if is_separator_row(cells):
                    continue
                rows.append(cells)
            if rows:
                text, methods, data_sources = classify_and_render_table(
                    current_bucket, rows
                )
                if text:
                    buckets[current_bucket].append(("table", text, methods, data_sources))
            continue

        cleaned = clean_bullet_or_note(raw_line)
        if cleaned:
            buckets[current_bucket].append(("text", cleaned, [], []))
        i += 1

    return buckets


# ----------------------------------------------------------------------------
# 텍스트 분할 (target/max/overlap 토큰 기준)
# ----------------------------------------------------------------------------

def split_into_units(text: str):
    """줄바꿈 두 번(문단) 우선, 그다음 줄바꿈 단위로 분할 단위를 만든다."""
    paragraphs = re.split(r"\n{2,}", text)
    units = []
    for p in paragraphs:
        if p.strip():
            units.append(p)
    return units if units else [text]


def chunk_text(text: str, target_tokens: int, max_tokens: int, overlap_tokens: int):
    """텍스트를 target/max 토큰 기준으로 나눈다. 전체가 max 이하면 그대로 반환."""
    total = count_tokens(text)
    if total <= max_tokens:
        return [text]

    units = split_into_units(text)
    # 문단 단위로도 못 나누면 줄 단위로 폴백.
    if len(units) <= 1:
        units = [ln for ln in text.split("\n") if ln.strip()]
    if not units:
        units = [text]

    parts = []
    current_units = []
    current_tokens = 0

    def flush():
        if current_units:
            parts.append("\n\n".join(current_units))

    for unit in units:
        u_tokens = count_tokens(unit)
        if current_units and current_tokens + u_tokens > max_tokens:
            flush()
            # overlap: 이전 청크 끝부분을 새 청크 시작에 포함.
            overlap_units = []
            overlap_count = 0
            for prev in reversed(current_units):
                pt = count_tokens(prev)
                if overlap_count + pt > overlap_tokens and overlap_units:
                    break
                overlap_units.insert(0, prev)
                overlap_count += pt
                if overlap_count >= overlap_tokens:
                    break
            current_units = list(overlap_units)
            current_tokens = overlap_count
        current_units.append(unit)
        current_tokens += u_tokens
        if current_tokens >= target_tokens and current_tokens >= max_tokens:
            flush()
            current_units = []
            current_tokens = 0

    flush()
    return parts if parts else [text]


# ----------------------------------------------------------------------------
# Parent / Child 빌드
# ----------------------------------------------------------------------------

DOCUMENT_ID = "seoul_core_projects_20260310"
DOCUMENT_VERSION = "2026-03-10"


def bucket_plain_text(entries):
    texts = []
    for kind, text, _, _ in entries:
        texts.append(text)
    return "\n".join(texts).strip()


def build_task_chunks(
    block: TaskBlock,
    catalog_entry: Optional[CatalogEntry],
    document_title: str,
    source_file: str,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    warnings: list,
    page_range_meta: Optional[dict] = None,
):
    task_no = block.task_no
    parent_id = f"{DOCUMENT_ID}_task_{task_no:03d}"

    field_name = catalog_entry.field if catalog_entry else ""
    core_business = catalog_entry.core_business if catalog_entry else block.related_business
    departments = catalog_entry.departments if catalog_entry else (
        [block.related_dept_text] if block.related_dept_text else []
    )
    catalog_project_name = catalog_entry.project_name if catalog_entry else block.project_name

    if catalog_entry and catalog_entry.project_name != block.project_name:
        warnings.append(
            {
                "type": "project_name_mismatch",
                "task_no": task_no,
                "catalog_project_name": catalog_entry.project_name,
                "body_project_name": block.project_name,
            }
        )

    buckets = segment_body(block)

    base_info_lines = [
        f"분야: {field_name}",
        f"핵심사업: {core_business}",
        f"연관사업: {block.related_business}",
        f"담당실국: {', '.join(departments)}",
    ]
    base_info_text = "\n".join(base_info_lines)

    # ---- Child별 본문 구성 ----
    policy_parts = [base_info_text]
    bg_text = bucket_plain_text(buckets["추진배경 및 목적"])
    if bg_text:
        policy_parts.append("## 추진배경 및 목적\n" + bg_text)
    policy_content = "\n\n".join(p for p in policy_parts if p).strip()

    main_text = bucket_plain_text(buckets["주요내용"])
    analysis_methods = []
    data_sources = []
    for kind, text, methods, sources in buckets["주요내용"]:
        analysis_methods.extend(methods)
        data_sources.extend(sources)
    analysis_content = ("## 주요내용\n" + main_text).strip() if main_text else ""

    results_text = bucket_plain_text(buckets["분석결과(예시)"])
    other_org_text = bucket_plain_text(buckets["타기관 분석사례"])
    results_parts = []
    if results_text:
        results_parts.append("## 분석결과(예시)\n" + results_text)
    if other_org_text:
        results_parts.append("## 타기관 분석사례\n" + other_org_text)
    results_content = "\n\n".join(results_parts).strip()

    usage_text = bucket_plain_text(buckets["실국 활용방안(예시)"])
    usage_content = ("## 실국 활용방안(예시)\n" + usage_text).strip() if usage_text else ""

    child_defs = [
        ("policy_context", policy_content),
        ("analysis_design", analysis_content),
        ("expected_results", results_content),
        ("administrative_use", usage_content),
    ]

    missing_required = []
    for chunk_type, content in child_defs:
        if not content and chunk_type != "expected_results":
            missing_required.append(chunk_type)
    if not results_content:
        # 분석결과 섹션 자체가 비면 required 누락으로 취급 (타기관 분석사례는 선택사항)
        if not bucket_plain_text(buckets["분석결과(예시)"]):
            missing_required.append("expected_results")

    # Parent 조립 시 기본정보 블록은 최상단에 한 번만 노출한다 (policy_context Child는
    # 독립 검색용으로 기본정보를 자체 포함하므로, Parent에서는 배경 섹션만 사용).
    policy_section_only = ("## 추진배경 및 목적\n" + bg_text).strip() if bg_text else ""
    parent_sections = [
        policy_section_only,
        analysis_content,
        results_content,
        usage_content,
    ]
    parent_body_parts = [
        f"[분석과제 {task_no}] {block.project_name}",
        "",
        base_info_text,
    ]
    for content in parent_sections:
        if content:
            parent_body_parts.append("")
            parent_body_parts.append(content)
    parent_content = "\n".join(parent_body_parts).strip()

    parent_record = {
        "id": parent_id,
        "parent_id": parent_id,
        "document_id": DOCUMENT_ID,
        "document_title": document_title,
        "document_version": DOCUMENT_VERSION,
        "source_file": source_file,
        "task_no": task_no,
        "field": field_name,
        "core_business": core_business,
        "project_name": block.project_name,
        "catalog_project_name": catalog_project_name,
        "related_business": block.related_business,
        "departments": departments,
        "chunk_type": "parent",
        "section_label": "분석과제 전체",
        "content": parent_content,
        "token_count": count_tokens(parent_content),
        "source_start_line": block.start_line + 1,
        "source_end_line": block.end_line,
    }
    if page_range_meta:
        parent_record.update(page_range_meta)

    child_records = []
    for chunk_type, content in child_defs:
        if not content:
            continue
        section_label = CHUNK_TYPE_LABELS[chunk_type]
        parts = chunk_text(content, target_tokens, max_tokens, overlap_tokens)
        part_count = len(parts)
        for part_idx, part_text in enumerate(parts, start=1):
            child_id = f"{parent_id}_{chunk_type}_{part_idx:02d}"
            header_lines = [
                f"문서명: {document_title}",
                f"분야: {field_name}",
                f"분석과제번호: {task_no}",
                f"분석과제명: {block.project_name}",
                f"연관사업: {block.related_business}",
                f"담당실국: {', '.join(departments)}",
                f"섹션: {section_label}",
            ]
            header_text = "\n".join(header_lines)
            embedding_text = header_text + "\n\n" + part_text

            record = {
                "id": child_id,
                "parent_id": parent_id,
                "document_id": DOCUMENT_ID,
                "document_title": document_title,
                "document_version": DOCUMENT_VERSION,
                "source_file": source_file,
                "task_no": task_no,
                "field": field_name,
                "core_business": core_business,
                "project_name": block.project_name,
                "catalog_project_name": catalog_project_name,
                "related_business": block.related_business,
                "departments": departments,
                "chunk_type": chunk_type,
                "section_label": section_label,
                "part_index": part_idx,
                "part_count": part_count,
                "content": part_text,
                "embedding_text": embedding_text,
                "token_count": count_tokens(part_text),
                "embedding_token_count": count_tokens(embedding_text),
                "analysis_methods": analysis_methods if chunk_type == "analysis_design" else [],
                "data_sources": data_sources if chunk_type == "analysis_design" else [],
                "source_start_line": block.start_line + 1,
                "source_end_line": block.end_line,
            }
            if page_range_meta:
                record.update(page_range_meta)
            child_records.append(record)

    return parent_record, child_records, missing_required


# ----------------------------------------------------------------------------
# 문서 개요 청크
# ----------------------------------------------------------------------------

def build_overview_chunks(
    catalog_entries, document_title, source_file, group_size=10, page_range_meta=None
):
    chunks = []
    sorted_entries = sorted(catalog_entries, key=lambda e: e.task_no)
    for group_idx in range(0, len(sorted_entries), group_size):
        group = sorted_entries[group_idx : group_idx + group_size]
        if not group:
            continue
        lo, hi = group[0].task_no, group[-1].task_no
        overview_no = group_idx // group_size + 1
        lines = [
            f"문서명: {document_title}",
            f"분석과제 범위: {lo}~{hi}",
            "",
        ]
        for e in group:
            lines.append(f"- 과제 {e.task_no}: {e.project_name}")
            lines.append(f"  분야: {e.field}")
            lines.append(f"  담당실국: {', '.join(e.departments)}")
            lines.append(f"  핵심사업: {e.core_business}")
        body = "\n".join(lines).strip()
        header = f"문서명: {document_title}\n분석과제 범위: {lo}~{hi}"
        chunk_id = f"{DOCUMENT_ID}_overview_{overview_no:02d}"
        record = {
            "id": chunk_id,
            "parent_id": None,
            "document_id": DOCUMENT_ID,
            "document_title": document_title,
            "document_version": DOCUMENT_VERSION,
            "source_file": source_file,
            "task_no": None,
            "field": None,
            "core_business": None,
            "project_name": None,
            "catalog_project_name": None,
            "related_business": None,
            "departments": [],
            "chunk_type": "overview",
            "section_label": "과제 목록 개요",
            "part_index": 1,
            "part_count": 1,
            "content": body,
            "embedding_text": body,
            "token_count": count_tokens(body),
            "embedding_token_count": count_tokens(body),
            "analysis_methods": [],
            "data_sources": [],
            "task_range": [lo, hi],
            "source_start_line": None,
            "source_end_line": None,
        }
        if page_range_meta:
            record.update(page_range_meta)
        chunks.append(record)
    return chunks


# ----------------------------------------------------------------------------
# 범위 파서
# ----------------------------------------------------------------------------

def parse_task_range(range_str, default_max):
    if not range_str:
        return 1, default_max
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", range_str)
    if not m:
        raise ValueError(f"--only-tasks 형식이 올바르지 않습니다: {range_str}")
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo < 1 or hi < lo:
        raise ValueError(f"--only-tasks 범위가 올바르지 않습니다: {range_str}")
    return lo, hi


# ----------------------------------------------------------------------------
# 검증
# ----------------------------------------------------------------------------

def run_validation(
    parents,
    children,
    overview_chunks,
    warnings,
    missing_required_map,
    expected_tasks,
    max_tokens,
    processed_task_nos,
):
    errors = []
    report_warnings = list(warnings)

    task_nos = sorted(processed_task_nos)
    if len(task_nos) != expected_tasks:
        errors.append(
            f"분석과제 수가 예상과 다릅니다: expected={expected_tasks}, actual={len(task_nos)}"
        )

    if task_nos:
        expected_set = set(range(task_nos[0], task_nos[0] + expected_tasks))
        missing = sorted(expected_set - set(task_nos))
        if missing:
            errors.append(f"누락된 과제번호: {missing}")

    dupes = [n for n in set(task_nos) if task_nos.count(n) > 1]
    if dupes:
        errors.append(f"중복된 과제번호: {sorted(dupes)}")

    if len(parents) != expected_tasks:
        errors.append(f"Parent 수가 예상과 다릅니다: expected={expected_tasks}, actual={len(parents)}")

    parent_ids = {p["id"] for p in parents}
    for c in children:
        if c["parent_id"] not in parent_ids:
            errors.append(f"유효하지 않은 parent_id를 가진 Child: {c['id']} -> {c['parent_id']}")

    task_child_count = {}
    for c in children:
        task_child_count[c["task_no"]] = task_child_count.get(c["task_no"], 0) + 1
    for tn in task_nos:
        if task_child_count.get(tn, 0) == 0:
            errors.append(f"Child가 하나도 없는 과제: {tn}")

    for c in children:
        if c["token_count"] > max_tokens:
            errors.append(
                f"Child가 max_tokens({max_tokens})를 초과합니다: {c['id']} ({c['token_count']} tokens)"
            )

    for c in children:
        if IMAGE_RE.search(c["embedding_text"]):
            errors.append(f"embedding_text에 이미지 경로가 남아있습니다: {c['id']}")
        if re.search(r"\|[\s:\-]*-{2,}[\s:\-]*\|", c["embedding_text"]):
            errors.append(f"embedding_text에 표 구분선이 남아있습니다: {c['id']}")

    for task_no, missing in missing_required_map.items():
        if missing:
            optional = {m for m in missing if m == "expected_results"}
            required_missing = [m for m in missing if m not in optional]
            if required_missing:
                errors.append(f"과제 {task_no}: 필수 섹션 누락 - {required_missing}")
            for m in optional:
                report_warnings.append(
                    {"type": "optional_section_missing", "task_no": task_no, "section": m}
                )

    return errors, report_warnings


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Markdown -> Parent/Child 청킹")
    parser.add_argument("md_path", help="입력 Markdown 파일 경로")
    parser.add_argument("--output-dir", default="data/chunks")
    parser.add_argument("--target-tokens", type=int, default=420)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--expected-tasks", type=int, default=None)
    parser.add_argument(
        "--only-tasks",
        default=None,
        help="처리할 분석과제 범위. 예: 1-10 (생략 시 문서 내 전체 과제 처리)",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--page-range-start",
        type=int,
        default=None,
        help="원본 인쇄 문서 기준 시작 페이지 번호 (지정 시 모든 청크에 page_range 메타데이터 부여)",
    )
    parser.add_argument(
        "--page-range-end",
        type=int,
        default=None,
        help="원본 인쇄 문서 기준 종료 페이지 번호",
    )
    args = parser.parse_args()

    page_range_meta = None
    if args.page_range_start is not None and args.page_range_end is not None:
        page_range_meta = {
            "page_range": f"{args.page_range_start}-{args.page_range_end}",
            "source_page_start": args.page_range_start,
            "source_page_end": args.page_range_end,
        }

    with open(args.md_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    lines = raw_text.split("\n")

    document_title = lines[0].strip()
    document_title = re.sub(r"^□\s*", "", document_title)
    source_file = os.path.basename(args.md_path)

    # 상단 과제 목록 범위: 헤더 행("| 연번 |")부터 첫 분석과제 표 직전까지.
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| 연번"):
            header_idx = i
            break
    if header_idx is None:
        print("오류: 상단 과제 목록 헤더를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(2)

    first_task_idx = None
    for i, line in enumerate(lines):
        if TITLE_BAR_RE.match(line.rstrip("\n")):
            first_task_idx = i
            break
    if first_task_idx is None:
        print("오류: 분석과제 블록을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(2)

    catalog_entries = parse_catalog(lines, header_idx, first_task_idx)
    catalog_by_no = {e.task_no: e for e in catalog_entries}

    all_blocks = find_task_blocks(lines)
    for b in all_blocks:
        extract_related_business(b)

    total_tasks_in_document = len(all_blocks)

    lo, hi = parse_task_range(args.only_tasks, total_tasks_in_document)
    selected_blocks = [b for b in all_blocks if lo <= b.task_no <= hi]
    selected_catalog = [e for e in catalog_entries if lo <= e.task_no <= hi]

    expected_tasks = args.expected_tasks
    if expected_tasks is None:
        expected_tasks = len(selected_blocks)

    warnings = []
    missing_required_map = {}
    parents = []
    children = []
    processed_task_nos = []

    for block in selected_blocks:
        catalog_entry = catalog_by_no.get(block.task_no)
        parent_record, child_records, missing_required = build_task_chunks(
            block,
            catalog_entry,
            document_title,
            source_file,
            args.target_tokens,
            args.max_tokens,
            args.overlap_tokens,
            warnings,
            page_range_meta,
        )
        parents.append(parent_record)
        children.extend(child_records)
        missing_required_map[block.task_no] = missing_required
        processed_task_nos.append(block.task_no)

    overview_chunks = build_overview_chunks(
        selected_catalog, document_title, source_file, page_range_meta=page_range_meta
    )

    all_embedding_chunks = children + overview_chunks

    validation_errors, report_warnings = run_validation(
        parents,
        children,
        overview_chunks,
        warnings,
        missing_required_map,
        expected_tasks,
        args.max_tokens,
        processed_task_nos,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    def write_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    parents_path = os.path.join(args.output_dir, "parents.jsonl")
    children_path = os.path.join(args.output_dir, "children.jsonl")
    overview_path = os.path.join(args.output_dir, "overview_chunks.jsonl")
    all_embedding_path = os.path.join(args.output_dir, "all_embedding_chunks.jsonl")
    csv_path = os.path.join(args.output_dir, "chunk_preview.csv")
    report_path = os.path.join(args.output_dir, "chunk_report.json")

    write_jsonl(parents_path, parents)
    write_jsonl(children_path, children)
    write_jsonl(overview_path, overview_chunks)
    write_jsonl(all_embedding_path, all_embedding_chunks)

    csv_columns = [
        "id",
        "parent_id",
        "task_no",
        "field",
        "project_name",
        "departments",
        "chunk_type",
        "section_label",
        "part_index",
        "part_count",
        "token_count",
        "embedding_token_count",
        "content_preview",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        for r in all_embedding_chunks:
            preview = (r.get("content") or "").replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "..."
            writer.writerow(
                [
                    r["id"],
                    r.get("parent_id") or "",
                    r.get("task_no") if r.get("task_no") is not None else "",
                    r.get("field") or "",
                    r.get("project_name") or "",
                    ", ".join(r.get("departments") or []),
                    r.get("chunk_type") or "",
                    r.get("section_label") or "",
                    r.get("part_index") or "",
                    r.get("part_count") or "",
                    r.get("token_count") or 0,
                    r.get("embedding_token_count") or 0,
                    preview,
                ]
            )

    # 입력 파일 자체에 처리 범위를 벗어난 분석과제 번호가 남아있는지 확인 (페이지 경계 누락 검증).
    out_of_range_tasks = sorted(
        {int(n) for n in re.findall(r"분석과제(\d+)", raw_text) if not (lo <= int(n) <= hi)}
    )
    if out_of_range_tasks:
        validation_errors.append(
            f"입력 파일에 처리 범위({lo}~{hi}) 밖의 분석과제 번호가 남아있습니다: {out_of_range_tasks}"
        )

    report = {
        "source_file": source_file,
        "document_title": document_title,
        "total_tasks_in_document": total_tasks_in_document,
        "requested_task_range": [lo, hi],
        "processed_tasks": len(processed_task_nos),
        "expected_tasks": expected_tasks,
        "parent_count": len(parents),
        "child_count": len(children),
        "overview_chunk_count": len(overview_chunks),
        "embedding_chunk_count": len(all_embedding_chunks),
        "token_method": TOKEN_METHOD,
        "target_tokens": args.target_tokens,
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "page_range": page_range_meta,
        "out_of_range_task_numbers_in_input": out_of_range_tasks,
        "validation_errors": validation_errors,
        "warnings": report_warnings,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"문서 제목: {document_title}")
    print(f"문서 내 전체 과제 수: {total_tasks_in_document}, 처리 범위: {lo}~{hi}")
    print(f"Parent: {len(parents)}, Child: {len(children)}, Overview: {len(overview_chunks)}")
    print(f"토큰 계산 방식: {TOKEN_METHOD}")
    print(f"경고: {len(report_warnings)}건, 검증 오류: {len(validation_errors)}건")
    print(f"출력 폴더: {os.path.abspath(args.output_dir)}")

    if args.strict and validation_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
