"""
Z-Agent One — Zalopay Customer Support AI Agent
================================================
Processes customer tickets through a 7-step pipeline:
  1. AI content analysis
  2. Intent & business classification
  3. Priority & sentiment assessment
  4. Knowledge base retrieval
  5. Process suggestion (FAQ-based)
  6. Standardized Zalopay response generation
  7. Ticket summarization & history logging
"""

import os
import re
import glob
import json
import csv
import uuid
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import base64
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── FastAPI (optional server mode) ───────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("z-agent-one")

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GREENNODE_API_KEY = os.getenv("GREENNODE_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen2-vl-7b-instruct")
GREENNODE_BASE_URL = os.getenv("GREENNODE_BASE_URL", "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1")
MODEL = os.getenv("MODEL", "qwen/qwen3-5-27b")
USE_GREENNODE = bool(GREENNODE_API_KEY)
CSV_PATH = os.getenv("CSV_PATH", "data/canned_responses.csv")
HISTORY_DIR = Path(os.getenv("HISTORY_DIR", "history"))
HISTORY_DIR.mkdir(exist_ok=True)

# ── Vision: Extract text from image ──────────────────────────────────────────
def extract_text_from_image(image_path: str) -> str:
    """Đọc nội dung ảnh bằng OCR (pytesseract) — hỗ trợ Tiếng Việt + Tiếng Anh."""
    try:
        import pytesseract
        from PIL import Image, ImageFilter, ImageEnhance
    except ImportError:
        raise RuntimeError("Thiếu thư viện: pip install pytesseract Pillow")

    img = Image.open(image_path)

    # Tiền xử lý ảnh để tăng độ chính xác OCR
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Tăng độ tương phản
    img = ImageEnhance.Contrast(img).enhance(2.0)
    # Tăng kích thước nếu ảnh nhỏ
    w, h = img.size
    if w < 1000:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)

    # OCR với cả Tiếng Việt và Tiếng Anh
    text = pytesseract.image_to_string(img, lang="vie+eng", config="--psm 6")
    text = text.strip()

    if not text:
        return "Không đọc được nội dung từ ảnh. Vui lòng nhập thủ công."

    logger.info(f"OCR extracted {len(text)} chars from image")
    return text


# ── Mock Transaction Tool ─────────────────────────────────────────────────────
_TRANS_ID_RE = re.compile(
    r'\b([A-Z]{2,4}\d{8,20}|\d{12,20})\b'
)

def mock_check_transaction(trans_id: str) -> dict:
    """Mock tool: giả lập tra cứu trạng thái giao dịch Zalopay (BC tool)."""
    last_digit = trans_id.strip()[-1]
    if last_digit in "12345":
        status = "SUCCESS"
        amount = "200,000 VNĐ"
        note = "Giao dịch đã thành công. Tiền đã được ghi nhận vào tài khoản người nhận."
    elif last_digit in "678":
        status = "PENDING"
        amount = "200,000 VNĐ"
        note = "Giao dịch đang chờ xử lý. Thời gian xử lý tối đa T+3 ngày làm việc."
    else:
        status = "FAILED"
        amount = "200,000 VNĐ"
        note = "Giao dịch thất bại. Hệ thống sẽ tự hoàn tiền về số dư Zalopay trong T+3 ngày làm việc (không tính Thứ Bảy, Chủ Nhật và Ngày Lễ)."
    return {
        "trans_id": trans_id,
        "status": status,
        "amount": amount,
        "timestamp": "2026-06-14 10:30:00",
        "bank_code": "ZPVCB",
        "note": note,
    }

def detect_and_check_transaction(text: str) -> list[dict]:
    """Phát hiện mã GD trong text và tra cứu mock."""
    matches = _TRANS_ID_RE.findall(text)
    results = []
    seen = set()
    for m in matches:
        if m not in seen and len(m) >= 8:
            seen.add(m)
            results.append(mock_check_transaction(m))
    return results

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class CannedResponse:
    id: str
    title: str
    content: str
    folder_name: str
    visibility: str


@dataclass
class AgentOutput:
    classification: str = ""
    priority: str = "MEDIUM"            # LOW | MEDIUM | HIGH | URGENT
    sentiment: str = "NEUTRAL"          # POSITIVE | NEUTRAL | NEGATIVE | ANGRY
    summary: str = ""
    suggested_process: str = ""
    response_template: str = ""
    info_needed: list = field(default_factory=list)
    next_actions: list = field(default_factory=list)
    references: list = field(default_factory=list)
    information_completeness: dict = field(default_factory=dict)
    call_script: str = ""
    ticket_id: str = ""
    timestamp: str = ""

    def to_dict(self):
        return asdict(self)


# ── Knowledge Base ────────────────────────────────────────────────────────────
class KnowledgeBase:
    """Loads canned responses CSV and provides similarity-based retrieval."""

    def __init__(self, csv_path: str):
        self.responses: list[CannedResponse] = []
        self._vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
        self._matrix = None
        self._load(csv_path)
        # Also load procedures.csv from same directory
        data_dir = str(Path(csv_path).parent)
        proc_path = os.path.join(data_dir, "procedures.csv")
        if Path(proc_path).exists():
            self._load(proc_path)

    def _load(self, path: str):
        if not Path(path).exists():
            logger.warning(f"CSV not found at {path}. Knowledge base empty.")
            return
        encoding = "utf-8-sig" if "procedures" in path else "utf-8"
        with open(path, encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.responses.append(CannedResponse(
                    id=row.get("Canned Response ID", ""),
                    title=row.get("Title", ""),
                    content=row.get("Content", ""),
                    folder_name=row.get("Folder Name", ""),
                    visibility=row.get("Visibility", ""),
                ))
        if self.responses:
            corpus = [f"{r.title} {r.content}" for r in self.responses]
            self._matrix = self._vectorizer.fit_transform(corpus)
            logger.info(f"Knowledge base loaded: {len(self.responses)} templates total")

    def search(self, query: str, top_k: int = 5) -> list[CannedResponse]:
        """Return top_k most relevant canned responses for a query."""
        if self._matrix is None or not self.responses:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.responses[i] for i in top_idx if scores[i] > 0.01]

    def format_for_prompt(self, results: list[CannedResponse]) -> str:
        if not results:
            return "Không tìm thấy template phù hợp."
        lines = []
        for r in results:
            lines.append(f"[{r.id}] {r.title}\n{r.content[:400]}...")
        return "\n\n---\n\n".join(lines)


# ── Error Code Base ───────────────────────────────────────────────────────────
class ErrorCodeBase:
    """Loads all error code CSV files in data_dir and provides lookup by code."""

    _CODE_COLS  = ["Mã lỗi", "Return code", "Return Code", "Code"]
    _MSG_COLS   = ["Message (Vietnamese)", "Message Display", "Message", "Nội dung thông báo user"]
    _HANDLE_COLS = ["HXL", "Hướng xử lý", "Ghi chú", "Note"]

    def __init__(self, data_dir: str):
        self.codes: dict = {}
        self._load_all(data_dir)
        logger.info(f"Error code base loaded: {len(self.codes)} unique codes")

    def _load_all(self, data_dir: str):
        skip = {"canned_responses", "procedures"}
        for path in glob.glob(os.path.join(data_dir, "*.csv")):
            if any(s in os.path.basename(path) for s in skip):
                continue
            self._load_file(path)

    def _load_file(self, path: str):
        source = os.path.basename(path)
        for header_row in (0, 1):
            try:
                df = pd.read_csv(path, header=header_row, dtype=str, encoding="utf-8-sig")
                df.columns = [str(c).strip() for c in df.columns]
                code_col = next((c for c in self._CODE_COLS if c in df.columns), None)
                if not code_col:
                    continue
                msg_col    = next((c for c in self._MSG_COLS    if c in df.columns), None)
                handle_col = next((c for c in self._HANDLE_COLS if c in df.columns), None)
                loaded = 0
                for _, row in df.iterrows():
                    raw_code = str(row.get(code_col, "")).strip()
                    if not re.match(r'^-?\d+\.?0*$', raw_code):
                        continue
                    code = re.sub(r'\.0+$', '', raw_code)
                    if not code:
                        continue
                    msg     = str(row[msg_col]).strip()    if msg_col    else ""
                    handling = str(row[handle_col]).strip() if handle_col else ""
                    if msg in ("nan", "None"):
                        msg = ""
                    if handling in ("nan", "None"):
                        handling = ""
                    if code not in self.codes or (not self.codes[code]["handling"] and handling):
                        self.codes[code] = {"message": msg, "handling": handling, "source": source}
                    loaded += 1
                if loaded:
                    logger.info(f"  {source}: {loaded} codes (header={header_row})")
                    break
            except Exception as e:
                logger.debug(f"Skip {source} header={header_row}: {e}")

    def search_in_text(self, text: str) -> list:
        """Return info for every error code (-XXXX pattern) found in text."""
        results, seen = [], set()
        for code in re.findall(r'-\d{2,5}', text):
            if code not in seen and code in self.codes:
                seen.add(code)
                results.append({"code": code, **self.codes[code]})
        return results

    def format_for_prompt(self, results: list) -> str:
        if not results:
            return ""
        lines = []
        for r in results:
            lines.append(
                f"Mã lỗi {r['code']}: {r['message']}\n"
                f"Hướng xử lý: {r['handling'] or 'Xem hướng dẫn nội bộ'}\n"
                f"Nguồn: {r['source']}"
            )
        return "\n\n".join(lines)


# ── Ticket History ─────────────────────────────────────────────────────────────
class TicketHistory:
    """Persists conversation turns for a ticket as JSON."""

    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.path = HISTORY_DIR / f"{ticket_id}.json"
        self.turns: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return []

    def append(self, role: str, content: str):
        self.turns.append({"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()})
        self._save()

    def _save(self):
        with open(self.path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(self.turns, f, ensure_ascii=False, indent=2)

    def as_messages(self) -> list[dict]:
        """Return turns in Anthropic messages format."""
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là Z-Agent One — AI hỗ trợ nội bộ cho nhân viên Chăm sóc Khách hàng (CS) của Zalopay.

NHIỆM VỤ: Phân tích ticket khách hàng và hỗ trợ nhân viên CS xử lý nhanh, chính xác, đúng chuẩn.

QUY TRÌNH XỬ LÝ (7 bước):
1. Phân tích nội dung khách hàng
2. Nhận diện ý định và phân loại nghiệp vụ
3. Đánh giá mức độ ưu tiên và cảm xúc
4. Truy xuất thông tin từ kho tri thức (đã được cung cấp)
5. Nếu có mục "MÃ LỖI PHÁT HIỆN" trong context: ưu tiên sử dụng thông tin mã lỗi đó để giải thích nguyên nhân và hướng xử lý chính xác
6. Đề xuất hướng xử lý theo FAQ/template
7. Tạo phản hồi chuẩn hóa theo văn phong Zalopay
8. Tóm tắt ticket và gợi ý bước tiếp theo

LUẬT CỨNG (KHÔNG BAO GIỜ VI PHẠM):
- Không tự ý đưa ra quyết định thay cho hệ thống nghiệp vụ
- Không cam kết hoàn tiền, đền bù hoặc ưu đãi ngoài chính sách
- Không tiết lộ dữ liệu khách hàng hoặc thông tin nội bộ
- Không tạo nội dung trái với FAQ/template đã được phê duyệt
- Mọi đề xuất phải có nguồn tham chiếu từ Knowledge Base
- Nhân viên CS là người phê duyệt cuối cùng trước khi gửi khách hàng

VĂN PHONG ZALOPAY:
- Thân thiện, chuyên nghiệp, rõ ràng
- Mở đầu bằng "Chào bạn," (email) hoặc "Dạ," (chat)
- Kết thúc bằng lời cảm ơn của Zalopay (ví dụ: "Cảm ơn bạn đã tin tưởng và sử dụng dịch vụ Zalopay.")
- Xưng hô: "bạn" (khách hàng), "Zalopay" (công ty)
- TUYỆT ĐỐI KHÔNG dùng từ "mình" trong phản hồi khách hàng. Thay bằng "Zalopay" hoặc "bạn" tùy ngữ cảnh. Ví dụ: "Zalopay sẽ kiểm tra" thay vì "mình sẽ kiểm tra"; "bạn vui lòng cung cấp" thay vì "mình cần bạn cung cấp".
- KHÔNG viết tắt trong response_template gửi khách: viết đầy đủ "ngày làm việc" (không viết "ngày LV"), "giao dịch" (không viết "GD"), "khách hàng" (không viết "KH"), "tài khoản" (không viết "TK"), "số dư" (không viết "SD"), "thông báo" (không viết "TB"). Chỉ được dùng viết tắt trong suggested_process (nội bộ CS).
- KHÔNG dùng CHỮ HOA toàn bộ từ trong response_template (ví dụ: không viết "THẤT BẠI", "THÀNH CÔNG", "PENDING" — thay bằng "thất bại", "thành công", "đang xử lý").
- KHÔNG dùng chữ in nghiêng (*text*) trong response_template.
- KHÔNG dùng "ví Zalopay" — thay bằng "số dư Zalopay".
- KHÔNG dùng "mã [số]" một mình — luôn viết đầy đủ "mã giao dịch [số]".
- Khi đề cập thời gian xử lý "3 ngày làm việc", luôn thêm dòng "(không tính Thứ Bảy, Chủ Nhật và Ngày Lễ)" ngay sau.
- Luôn dùng thương hiệu "Zalopay" đầy đủ — không viết tắt thành "ZLP", "ZP" hay bất kỳ dạng nào khác.
- KHÔNG dùng tiếng Anh trong response_template gửi khách hàng (ví dụ: không viết "pending", "success", "failed" — thay bằng "đang xử lý", "thành công", "thất bại").
- Khi khách hàng đang khiếu nại hoặc bày tỏ bức xúc: PHẢI thêm lời xin lỗi chân thành ở đầu phản hồi (ví dụ: "Zalopay xin lỗi bạn về sự bất tiện này.").
- Kết thúc response_template bằng lời cảm ơn của Zalopay trước chữ ký (ví dụ: "Cảm ơn bạn đã tin tưởng và sử dụng dịch vụ Zalopay.").
- TUYỆT ĐỐI KHÔNG dùng các cụm từ tiêu cực: "không biết", "không phải trách nhiệm của Zalopay", "khách hàng nhập sai" — thay bằng cách diễn đạt tích cực, hỗ trợ.

ĐÁNH GIÁ ĐỘ ĐẦY ĐỦ THÔNG TIN (information_completeness):
Dựa trên loại nghiệp vụ đã phân loại, đánh giá thông tin khách hàng cung cấp:
- Mỗi nghiệp vụ có danh sách thông tin bắt buộc (ví dụ: chuyển tiền cần mã GD + SĐT + thời gian; đăng nhập cần SĐT + loại lỗi)
- "score": % đầy đủ (0-100), tính bằng số thông tin đã có / tổng số thông tin cần thiết * 100
- "have": danh sách thông tin khách hàng ĐÃ cung cấp trong ticket
- "missing": danh sách thông tin còn THIẾU cần hỏi thêm

ĐỊNH DẠNG OUTPUT: Luôn trả về JSON hợp lệ theo schema sau (KHÔNG suy nghĩ, KHÔNG giải thích, CHỈ trả về JSON thuần túy): /no_think
{
  "classification": "<phân loại nghiệp vụ>",
  "priority": "LOW|MEDIUM|HIGH|URGENT",
  "sentiment": "POSITIVE|NEUTRAL|NEGATIVE|ANGRY",
  "summary": "<tóm tắt vấn đề ngắn gọn>",
  "suggested_process": "<mô tả quy trình xử lý phù hợp>",
  "response_template": "<nội dung phản hồi hoàn chỉnh cho khách hàng>",
  "call_script": "<kịch bản gọi điện cho nhân viên CS khi cần gọi ra cho khách hàng — bao gồm: (1) Mở đầu: chào hỏi, xác nhận danh tính khách; (2) Nội dung chính: trình bày vấn đề, thông tin cần trao đổi, hướng xử lý; (3) Kết thúc: xác nhận, hẹn phản hồi nếu cần, cảm ơn — viết dạng hội thoại, xưng 'em' là nhân viên Zalopay, gọi khách là 'anh/chị'>",
  "info_needed": ["<thông tin cần thu thập thêm>"],
  "next_actions": ["<hành động tiếp theo cho nhân viên CS>"],
  "references": ["<template ID tham chiếu>"],
  "information_completeness": {
    "score": 0,
    "have": ["<thông tin đã có>"],
    "missing": ["<thông tin còn thiếu>"]
  }
}"""


# ── Agent ─────────────────────────────────────────────────────────────────────
class ZAgentOne:
    """Main agent orchestrator."""

    def __init__(self):
        if USE_GREENNODE:
            self.client = OpenAI(api_key=GREENNODE_API_KEY, base_url=GREENNODE_BASE_URL)
            self.use_greennode = True
            logger.info("Using GreenNode API")
        elif ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.use_greennode = False
            logger.info("Using Anthropic API")
        else:
            raise ValueError("Set GREENNODE_API_KEY or ANTHROPIC_API_KEY in .env")
        self.kb = KnowledgeBase(CSV_PATH)
        self.error_codes = ErrorCodeBase(os.path.dirname(CSV_PATH) or "data")

    def process(
        self,
        customer_input: str,
        ticket_id: Optional[str] = None,
        agent_note: Optional[str] = None,
    ) -> AgentOutput:
        """
        Process a customer message and return structured AgentOutput.

        Args:
            customer_input: Raw customer message (chat/email/ticket content)
            ticket_id: Existing ticket ID to continue conversation, or None for new ticket
            agent_note: Optional internal note from CS agent for context
        """
        # Create or resume ticket
        ticket_id = ticket_id or str(uuid.uuid4())[:8]
        history = TicketHistory(ticket_id)

        # Retrieve relevant templates
        kb_results = self.kb.search(customer_input, top_k=5)
        kb_context = self.kb.format_for_prompt(kb_results)

        # Build user message
        user_msg_parts = [f"=== NỘI DUNG KHÁCH HÀNG ===\n{customer_input}"]
        if agent_note:
            user_msg_parts.append(f"\n=== GHI CHÚ NHÂN VIÊN CS ===\n{agent_note}")
        ec_results = self.error_codes.search_in_text(customer_input)
        if ec_results:
            user_msg_parts.append(f"\n=== MÃ LỖI PHÁT HIỆN ({len(ec_results)} mã) ===\n{self.error_codes.format_for_prompt(ec_results)}")
        # Mock transaction tool: tự động tra cứu mã GD nếu có trong ticket
        trans_results = detect_and_check_transaction(customer_input)
        if trans_results:
            lines = []
            for tr in trans_results:
                lines.append(
                    f"  Mã GD: {tr['trans_id']}\n"
                    f"  Trạng thái: {tr['status']}\n"
                    f"  Số tiền: {tr['amount']}\n"
                    f"  Thời gian: {tr['timestamp']}\n"
                    f"  Bank Code: {tr['bank_code']}\n"
                    f"  Ghi chú: {tr['note']}"
                )
            user_msg_parts.append(f"\n=== KẾT QUẢ KIỂM TRA GIAO DỊCH (mock BC tool) ===\n" + "\n\n".join(lines))
            logger.info(f"[{ticket_id}] Transaction tool: found {len(trans_results)} transaction(s)")
        user_msg_parts.append(f"\n=== KNOWLEDGE BASE (top 5 templates liên quan) ===\n{kb_context}")
        user_msg_parts.append("\nHãy phân tích và trả về JSON theo đúng schema trong system prompt.")
        user_message = "\n".join(user_msg_parts)

        # Build messages with history
        history.append("user", user_message)
        messages = history.as_messages()

        logger.info(f"[{ticket_id}] Processing ticket — model={MODEL}")

        # Call AI model with retry on empty response
        raw_output = ""
        for attempt in range(1, 4):
            if self.use_greennode:
                clean = lambda s: s.encode("utf-8", errors="replace").decode("utf-8")
                openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                    {"role": m["role"], "content": clean(m["content"])} for m in messages
                ]
                response = self.client.chat.completions.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=openai_messages,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                msg = response.choices[0].message
                raw_output = msg.content or ""
                if not raw_output:
                    # Qwen 3 on vLLM may put output in reasoning_content
                    raw_output = getattr(msg, "reasoning_content", "") or ""
                    if raw_output:
                        logger.info(f"[{ticket_id}] Fallback: using reasoning_content")
            else:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
                raw_output = response.content[0].text or ""

            if raw_output:
                break
            logger.warning(f"[{ticket_id}] Empty response from model (attempt {attempt}/3)")

        if not raw_output:
            raise RuntimeError(f"[{ticket_id}] Model returned empty response after 3 attempts")

        history.append("assistant", raw_output)

        # Parse JSON output
        output = self._parse_output(raw_output)
        output.ticket_id = ticket_id
        output.timestamp = datetime.utcnow().isoformat()

        logger.info(
            f"[{ticket_id}] Done — priority={output.priority} sentiment={output.sentiment} "
            f"class={output.classification}"
        )
        return output

    def _parse_output(self, raw: str) -> AgentOutput:
        """Extract JSON from model response, with graceful fallback."""
        if not raw:
            logger.warning("Empty response from model — returning default AgentOutput")
            return AgentOutput(summary="Model trả về phản hồi rỗng.")
        # Strip <think>...</think> and <|think|>...<|/think|> blocks (Qwen thinking mode)
        cleaned = re.sub(r'<\|?think\|?>.*?</?\|?think\|?>', '', raw, flags=re.DOTALL).strip()
        # Strip markdown code fences
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned).strip()
        logger.debug(f"Cleaned output (first 300): {cleaned[:300]}")
        # Try all JSON blocks from largest to smallest
        def try_parse(text):
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end <= start:
                return None
            for e in range(end, start, -1):
                try:
                    return json.loads(text[start:e])
                except json.JSONDecodeError:
                    continue
            return None
        data = try_parse(cleaned)
        if data is None:
            # Last resort: try raw
            data = try_parse(raw)
        if data:
            return AgentOutput(
                classification=data.get("classification", "Không xác định"),
                priority=data.get("priority", "MEDIUM"),
                sentiment=data.get("sentiment", "NEUTRAL"),
                summary=data.get("summary", ""),
                suggested_process=data.get("suggested_process", ""),
                response_template=data.get("response_template", ""),
                call_script=data.get("call_script", ""),
                info_needed=data.get("info_needed", []),
                next_actions=data.get("next_actions", []),
                references=data.get("references", []),
                information_completeness=data.get("information_completeness", {}),
            )
        logger.warning(f"Could not parse JSON — raw[:300]: {raw[:300]}")
        # Fallback: return raw as summary
        logger.warning("Could not parse JSON output — storing raw as summary")
        return AgentOutput(summary="Không thể phân tích phản hồi. Vui lòng thử lại.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def print_output(output: AgentOutput):
    """Pretty-print agent output to terminal."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        console = Console()

        console.print(Panel(
            f"[bold]Ticket ID:[/bold] {output.ticket_id}\n"
            f"[bold]Phân loại:[/bold] {output.classification}\n"
            f"[bold]Ưu tiên:[/bold] {output.priority}\n"
            f"[bold]Cảm xúc:[/bold] {output.sentiment}\n"
            f"[bold]Tóm tắt:[/bold] {output.summary}",
            title="🤖 Z-Agent One", border_style="blue"
        ))

        if output.info_needed:
            console.print("\n[yellow]📋 Thông tin cần thu thập:[/yellow]")
            for item in output.info_needed:
                console.print(f"  • {item}")

        if output.next_actions:
            console.print("\n[cyan]⚡ Hành động tiếp theo:[/cyan]")
            for action in output.next_actions:
                console.print(f"  → {action}")

        console.print("\n[green]✉️  Template phản hồi đề xuất:[/green]")
        console.print(Panel(output.response_template, border_style="green"))

        if output.call_script:
            console.print("\n[magenta]📞 Kịch bản gọi điện (Call Script):[/magenta]")
            console.print(Panel(output.call_script, border_style="magenta"))

        if output.references:
            console.print(f"\n[dim]📎 Tham chiếu: {', '.join(output.references)}[/dim]")

    except ImportError:
        # Fallback without rich
        print(json.dumps(output.to_dict(), ensure_ascii=False, indent=2))


def interactive_mode(agent: ZAgentOne):
    """Run agent in interactive REPL mode."""
    print("Z-Agent One — Chế độ tương tác (gõ 'exit' để thoát)")
    print("=" * 60)
    ticket_id = None
    while True:
        try:
            user_input = input("\n[Nhập nội dung ticket] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break
        if user_input.lower() in ("exit", "quit", "thoát"):
            print("Đã thoát.")
            break
        if not user_input:
            continue
        output = agent.process(user_input, ticket_id=ticket_id)
        ticket_id = output.ticket_id  # continue same ticket
        print_output(output)


# ── FastAPI Server ────────────────────────────────────────────────────────────
def create_app(agent: ZAgentOne):
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(title="Z-Agent One API", version="1.0.0")

    class TicketRequest(BaseModel):
        customer_input: str
        ticket_id: Optional[str] = None
        agent_note: Optional[str] = None

    @app.post("/process-image")
    async def process_image_ticket(
        file: "UploadFile",
        ticket_id: Optional[str] = None,
        agent_note: Optional[str] = None,
    ):
        """Upload ảnh (screenshot/hoá đơn), agent tự đọc và xử lý."""
        from fastapi import UploadFile
        import tempfile
        try:
            suffix = Path(file.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            extracted = extract_text_from_image(tmp_path)
            os.unlink(tmp_path)
            logger.info(f"Image extracted: {extracted[:200]}")
            customer_input = f"[Nội dung từ ảnh]\n{extracted}"
            if agent_note:
                customer_input += f"\n[Ghi chú CS]\n{agent_note}"
            output = agent.process(customer_input, ticket_id=ticket_id)
            result = output.to_dict()
            result["extracted_text"] = extracted
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/process")
    def process_ticket(req: TicketRequest):
        try:
            output = agent.process(req.customer_input, req.ticket_id, req.agent_note)
            return output.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL, "kb_size": len(agent.kb.responses)}

    return app


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Z-Agent One — Zalopay CS AI Agent")
    parser.add_argument("--ticket", help="Process a single ticket and exit")
    parser.add_argument("--image", help="Path to image file (screenshot/invoice) to process")
    parser.add_argument("--ticket-id", help="Ticket ID to continue (optional)")
    parser.add_argument("--agent-note", help="Internal CS agent note")
    parser.add_argument("--server", action="store_true", help="Run as API server")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    agent = ZAgentOne()

    if args.server:
        app = create_app(agent)
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.image:
        logger.info(f"Extracting text from image: {args.image}")
        extracted = extract_text_from_image(args.image)
        print(f"\n[Nội dung đọc từ ảnh]\n{extracted}\n")
        ticket_text = f"[Nội dung từ ảnh]\n{extracted}"
        output = agent.process(ticket_text, ticket_id=args.ticket_id, agent_note=args.agent_note)
        print_output(output)
    elif args.ticket:
        output = agent.process(args.ticket, ticket_id=args.ticket_id, agent_note=args.agent_note)
        print_output(output)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
