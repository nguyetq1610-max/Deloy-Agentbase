# Z-Agent One

Z-Agent One là nền tảng AI hỗ trợ Chăm sóc Khách hàng (Customer Service Intelligence Platform) được phát triển nhằm nâng cao hiệu quả vận hành và chất lượng dịch vụ tại ZaloPay.

Khác với chatbot hoặc công cụ tìm kiếm template truyền thống, Z-Agent One hoạt động như một AI Copilot dành cho nhân viên CS, hỗ trợ từ phân tích yêu cầu, truy xuất tri thức, đề xuất hướng xử lý đến tạo phản hồi chuẩn hóa.

### Tính năng nổi bật

- Nhận diện và phân loại nghiệp vụ tự động.
- Tóm tắt nội dung khách hàng.
- Xác định thông tin cần thu thập.
- Truy xuất Knowledge Base (FAQ, chính sách, quy trình, template).
- Đề xuất Next Best Action.
- Sinh phản hồi theo chuẩn CS ZaloPay.
- Hỗ trợ phát hiện khoảng trống tri thức để cải tiến Knowledge Base.

### Giá trị mang lại

- Giảm thời gian xử lý ticket.
- Chuẩn hóa chất lượng phản hồi.
- Giảm thiểu rủi ro tư vấn sai, thiếu thông tin hoặc không nhất quán.
- Đảm bảo hướng xử lý luôn dựa trên Knowledge Base đã được phê duyệt.
- Rút ngắn thời gian đào tạo nhân viên mới.
- Tăng năng suất vận hành và nâng cao trải nghiệm khách hàng.

### Tầm nhìn

Z-Agent One không chỉ là trợ lý AI hỗ trợ trả lời khách hàng mà còn là nền tảng giúp đội ngũ CS khai thác tri thức hiệu quả hơn, ra quyết định nhanh hơn và liên tục cải thiện chất lượng dịch vụ thông qua dữ liệu thực tế.

---

## Kiến trúc

```
Customer Ticket
      │
      ▼
┌─────────────────────────────────┐
│         Z-Agent One             │
│                                 │
│  ┌──────────┐  ┌─────────────┐  │
│  │    KB    │  │ Error Codes │  │
│  │ 1104 tmpl│  │  393 codes  │  │
│  └──────────┘  └─────────────┘  │
│                                 │
│  ┌──────────────────────────┐   │
│  │   Qwen3-5-27B (GreenNode)│   │
│  │   via OpenAI-compatible  │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
      │
      ▼
AgentOutput: classification, priority, sentiment,
             summary, response_template, call_script,
             next_actions, info_needed, references
```

## Cài đặt & Chạy

### Yêu cầu
- Docker
- GreenNode API Key (hoặc Anthropic API Key)

### Chạy với Docker

```bash
# Clone repo
git clone <repo-url>
cd z-agent-one

# Tạo file .env
cp .env.example .env
# Điền GREENNODE_API_KEY vào .env

# Build & Run
docker build -t z-agent-one .
docker run -it --name z-agent-one \
  --env-file .env \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/history:/app/history" \
  z-agent-one
```

### Chạy API Server

```bash
docker run -d --name z-agent-one \
  --env-file .env \
  -p 8000:8080 \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/history:/app/history" \
  z-agent-one python agent.py --server
```

### Xử lý ticket đơn

```bash
docker exec z-agent-one python agent.py \
  --ticket "Mã giao dịch 240614123456783, khách hàng báo quét mã VietQR thành công nhưng đơn hàng chưa xác nhận"
```

### Hỏi tiếp cùng ticket

```bash
docker exec z-agent-one python agent.py \
  --ticket-id "TICKET_ID" \
  --ticket "Khách hỏi thêm: tiền có bị trừ không?"
```

## Cấu trúc thư mục

```
z-agent-one/
├── agent.py              # Agent chính
├── Dockerfile
├── requirements.txt
├── .env.example
└── data/
    ├── canned_responses.csv     # 1090 template chuẩn
    ├── procedures.csv           # 14 quy trình nội bộ CS
    ├── ErrorCode_Final.csv      # Mã lỗi hệ thống
    └── ...
```

## API

### POST /process

```json
{
  "customer_input": "Nội dung ticket khách hàng",
  "ticket_id": "optional — để tiếp tục hội thoại",
  "agent_note": "optional — ghi chú nội bộ từ CS"
}
```

**Response:**
```json
{
  "classification": "VietQR - Đơn hàng chưa xác nhận",
  "priority": "MEDIUM",
  "sentiment": "NEGATIVE",
  "summary": "Khách thanh toán VietQR thành công nhưng đơn hàng chưa được xác nhận",
  "suggested_process": "...",
  "response_template": "Chào bạn, ...",
  "call_script": "Em chào anh/chị, ...",
  "info_needed": [],
  "next_actions": ["Xác nhận trạng thái đơn với merchant"],
  "references": ["PROC_VIETQR_003"],
  "information_completeness": { "score": 80, "have": [...], "missing": [...] }
}
```

## Team

| Thành viên | Bộ phận | Email |
|---|---|---|
| Super Bunny | ZaloPay CS | quetn@vng.com.vn |
