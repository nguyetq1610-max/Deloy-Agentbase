"""
Z-Agent One — ZaloPay Customer Support AI Agent
================================================
Processes customer tickets through a 7-step pipeline:
  1. AI content analysis
  2. Intent & business classification
  3. Priority & sentiment assessment
  4. Knowledge base retrieval
  5. Process suggestion (FAQ-based)
  6. Standardized ZaloPay response generation
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
GREENNODE_BASE_URL = os.getenv("GREENNODE_BASE_URL", "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1")
MODEL = os.getenv("MODEL", "qwen/qwen3-5-27b")
USE_GREENNODE = bool(GREENNODE_API_KEY)
CSV_PATH = os.getenv("CSV_PATH", "data/canned_responses.csv")
HISTORY_DIR = Path(os.getenv("HISTORY_DIR", "history"))
HISTORY_DIR.mkdir(exist_ok=True)

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

    def _load(self, path: str):
        if not Path(path).exists():
            logger.warning(f"CSV not found at {path}. Knowledge base empty.")
            return
        with open(path, encoding="utf-8") as f:
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
            logger.info(f"Knowledge base loaded: {len(self.responses)} templates")

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
        for path in glob.glob(os.path.join(data_dir, "*.csv")):
            if "canned_responses" in os.path.basename(path):
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
SYSTEM_PROMPT = """Bạn là Z-Agent One — AI hỗ trợ nội bộ cho nhân viên Chăm sóc Khách hàng (CS) của ZaloPay.

NHIỆM VỤ: Phân tích ticket khách hàng và hỗ trợ nhân viên CS xử lý nhanh, chính xác, đúng chuẩn.

QUY TRÌNH XỬ LÝ (7 bước):
1. Phân tích nội dung khách hàng
2. Nhận diện ý định và phân loại nghiệp vụ
3. Đánh giá mức độ ưu tiên và cảm xúc
4. Truy xuất thông tin từ kho tri thức (đã được cung cấp)
5. Nếu có mục "MÃ LỖI PHÁT HIỆN" trong context: ưu tiên sử dụng thông tin mã lỗi đó để giải thích nguyên nhân và hướng xử lý chính xác
6. Đề xuất hướng xử lý theo FAQ/template
7. Tạo phản hồi chuẩn hóa theo văn phong ZaloPay
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
- Kết thúc với chữ ký chuẩn: ZALOPAY - ỨNG DỤNG THANH TOÁN MỌI DỊCH VỤ
- Xưng hô: "bạn" (khách hàng), "Zalopay" (công ty)

ĐÁNH GIÁ ĐỘ ĐẦY ĐỦ THÔNG TIN (information_completeness):
Dựa trên loại nghiệp vụ đã phân loại, đánh giá thông tin khách hàng cung cấp:
- Mỗi nghiệp vụ có danh sách thông tin bắt buộc (ví dụ: chuyển tiền cần mã GD + SĐT + thời gian; đăng nhập cần SĐT + loại lỗi)
- "score": % đầy đủ (0-100), tính bằng số thông tin đã có / tổng số thông tin cần thiết * 100
- "have": danh sách thông tin khách hàng ĐÃ cung cấp trong ticket
- "missing": danh sách thông tin còn THIẾU cần hỏi thêm

ĐỊNH DẠNG OUTPUT: Luôn trả về JSON hợp lệ theo schema sau:
{
  "classification": "<phân loại nghiệp vụ>",
  "priority": "LOW|MEDIUM|HIGH|URGENT",
  "sentiment": "POSITIVE|NEUTRAL|NEGATIVE|ANGRY",
  "summary": "<tóm tắt vấn đề ngắn gọn>",
  "suggested_process": "<mô tả quy trình xử lý phù hợp>",
  "response_template": "<nội dung phản hồi hoàn chỉnh cho khách hàng>",
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
                )
                raw_output = response.choices[0].message.content or ""
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
        # Try to extract JSON block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end])
                return AgentOutput(
                    classification=data.get("classification", "Không xác định"),
                    priority=data.get("priority", "MEDIUM"),
                    sentiment=data.get("sentiment", "NEUTRAL"),
                    summary=data.get("summary", ""),
                    suggested_process=data.get("suggested_process", ""),
                    response_template=data.get("response_template", ""),
                    info_needed=data.get("info_needed", []),
                    next_actions=data.get("next_actions", []),
                    references=data.get("references", []),
                    information_completeness=data.get("information_completeness", {}),
                )
            except json.JSONDecodeError:
                pass
        # Fallback: return raw as summary
        logger.warning("Could not parse JSON output — storing raw as summary")
        return AgentOutput(summary=raw)


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
    parser = argparse.ArgumentParser(description="Z-Agent One — ZaloPay CS AI Agent")
    parser.add_argument("--ticket", help="Process a single ticket and exit")
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
    elif args.ticket:
        output = agent.process(args.ticket, ticket_id=args.ticket_id, agent_note=args.agent_note)
        print_output(output)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
