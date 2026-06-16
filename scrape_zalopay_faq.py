#!/usr/bin/env python3
"""
ZaloPay FAQ Scraper — Z-Agent One Knowledge Base Builder
=========================================================
Scrapes all articles from zalopay.vn/hoi-dap and saves to:
  data/faq_zalopay.csv

Then you merge faq_zalopay.csv into data/canned_responses.csv and rebuild Docker.

Requirements:
  pip install requests beautifulsoup4

Usage:
  python3 scrape_zalopay_faq.py
"""

import requests
import csv
import re
import time
import os
import unicodedata
from bs4 import BeautifulSoup
from datetime import datetime

BASE_URL    = "https://zalopay.vn"
OUTPUT_CSV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faq_zalopay.csv")
DELAY       = 0.8   # seconds between requests (be polite)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.5",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─── Known article list for "Quản lý Tài khoản" (extracted from index page) ───
# Format: (category_slug, subcategory_slug, [article_titles...])
KNOWN_ARTICLES = [
    ("quan-ly-tai-khoan", "khoa-mo-khoa-tai-khoan", [
        "Làm thế nào để đóng tài khoản Ví điện tử Zalopay?",
        "Cách khoá tài khoản Zalopay",
        "Cách mở khoá tài khoản Zalopay",
    ]),
    ("quan-ly-tai-khoan", "dinh-danh-tai-khoan", [
        "Gửi yêu cầu định danh nhưng chưa được duyệt",
        "Cách định danh tài khoản Zalopay",
        "Gặp lỗi khi gửi yêu cầu định danh",
        "Tại sao tôi không thể sử dụng số dư tài khoản Zalopay để thực hiện giao dịch?",
        "Điều chỉnh thông tin định danh",
        "Cách thực hiện sinh trắc học bằng VNeID khi điện thoại không hỗ trợ NFC hoặc sinh trắc học",
        "Tôi chưa thực hiện định danh thì có sử dụng được các dịch vụ của Zalopay không?",
        "Cập nhật thông tin cá nhân với Zalopay có an toàn không?",
        "Hướng dẫn cách cập nhật hình ảnh giấy tờ tùy thân và hình chân dung lên ứng dụng Zalopay",
        "Có thể sử dụng giấy tờ nào để định danh?",
        "Thời gian trả kết quả định danh là bao lâu?",
    ]),
    ("quan-ly-tai-khoan", "mat-khau-thanh-toan", [
        "Cách xử lý khi quên mật khẩu thanh toán",
        "Khi nào nên đổi mật khẩu thanh toán?",
        "Bật/tắt tính năng bảo mật vân tay",
        "Cách đổi mật khẩu thanh toán?",
        "Mật khẩu thanh toán là gì?",
        "Tôi bị mất số điện thoại nên không nhận được mật khẩu thanh toán",
    ]),
    ("quan-ly-tai-khoan", "huong-dan-dang-ky-va-su-dung-tai-khoan", [
        "Hướng dẫn đổi số điện thoại khi quên mật khẩu thanh toán",
        "Có thể thay đổi số điện thoại đăng ký tài khoản Zalopay không?",
        "Không nhận được mã OTP gửi về số điện thoại Zalopay",
        "Cách tạo tài khoản Zalo",
        "Chuyển nhượng tài khoản Zalo có ảnh hưởng đến tài khoản Zalopay không?",
        "Đăng ký tài khoản Zalopay nhưng bị báo 'Số điện thoại đã được đăng ký'",
        "Tôi mới mua một số điện thoại và nhận được thông báo số điện thoại đã được đăng ký Zalopay, tôi cần làm gì?",
        "Có thể đăng nhập nhiều tài khoản Zalopay trên 1 thiết bị được không?",
        "Cách xử lý khi không đăng nhập được tài khoản Zalopay",
        "Cách đăng nhập tài khoản Zalopay",
        "Cách tạo tài khoản Zalopay",
        "Cách đăng xuất tài khoản Zalopay",
        "Nếu xóa tài khoản Zalo thì có thể sử dụng được Zalopay không?",
        "Số điện thoại đăng ký Zalopay có cập nhật theo số điện thoại Zalo không?",
        "Làm thế nào để lấy thông tin xác nhận chủ thuê bao?",
        "Tôi không gửi được yêu cầu đổi số điện thoại",
        "Chưa có tài khoản Zalo thì có sử dụng Zalopay được không?",
        "Cách lấy thông tin xác nhận chủ thuê bao",
        "Có thể đăng nhập 1 tài khoản Zalopay trên nhiều thiết bị được không?",
        "Mua SIM mới nhưng số điện thoại có sẵn tài khoản Zalopay",
    ]),
    ("quan-ly-tai-khoan", "diem-tin-cay", [
        "Điểm tin cậy là gì?",
        "Lợi ích khi tăng điểm tin cậy là gì?",
        "Sau bao lâu sẽ được cập nhật điểm tin cậy?",
        "Làm sao để tôi tăng điểm?",
        "Các hành vi có thể làm giảm điểm",
    ]),
    ("quan-ly-tai-khoan", "cac-van-de-thuong-gap-ve-tai-khoan", [
        "Không rút tiền được trong số dư ví do chưa xác thực tài khoản",
        "Tôi không thể thêm phương thức thanh toán Zalopay trên ứng dụng của đối tác",
        "Tôi có thể sử dụng được các dịch vụ của Zalopay mà không cần định danh tài khoản không?",
        "Tôi chưa liên kết ngân hàng thì có sử dụng được Zalopay không",
        "Tôi đã xác thực sinh trắc học nhưng vẫn không sử dụng được dịch vụ của Zalopay",
        "Tôi đã xác thực tài khoản nhưng vẫn không chọn được nguồn tiền từ số dư ví",
        "Tôi không chọn được nguồn tiền từ Số dư ví hay nguồn tiền từ Ngân hàng liên kết",
        "Tôi có thể sử dụng được các dịch vụ của Zalopay mà không cần xác thực sinh trắc học (NFC) không?",
        "Tôi không thể thanh toán bằng Số dư ví và Ngân hàng liên kết",
    ]),
]

# ─── Remaining categories — articles discovered dynamically ───────────────────
DISCOVER_CATEGORIES = [
    ("nap-tien-rut-tien",       ["rut-tien", "nap-tien"]),
    ("chuyen-tien-nhan-tien",   ["nhac-chuyen-tien", "chuyen-tien-den-vi-khac",
                                  "chuyen-tien-den-the-tai-khoan-ngan-hang",
                                  "nhan-tien", "nhan-tien-quoc-te"]),
    ("thanh-toan-dich-vu",      ["metro", "bao-hiem-suc-khoe-24-7", "ve-tau-hoa",
                                  "ve-tham-quan", "ve-xe-khach", "ve-may-bay",
                                  "khach-san", "thanh-toan-hoa-don-vay-tieu-dung",
                                  "thanh-toan-hoc-phi", "thanh-toan-bao-hiem-hoa-don",
                                  "thanh-toan-hoa-don-dien-nuoc-internet-truyen-hinh",
                                  "thuong-mai-dien-tu", "thanh-toan-dien-thoai-tra-sau",
                                  "goi-cuoc-combo", "tra-no-the-tin-dung",
                                  "cac-dich-vu-khac", "ve-phim", "phi-dich-vu",
                                  "ma-thanh-toan", "khieu-nai-thanh-toan",
                                  "nap-3g-4g", "the-dien-thoai", "nap-tien-dien-thoai",
                                  "esim-du-lich"]),
    ("an-toan-va-bao-mat",      ["giai-phap-bao-mat-cua-zalopay",
                                  "canh-bao-cac-tinh-huong-lua-dao-pho-bien",
                                  "hoi-dap-ve-an-toan-bao-mat",
                                  "bien-phap-tang-cuong-bao-mat-tai-khoan",
                                  "huong-dan-lien-he-khan-cap"]),
    ("lien-ket-ngan-hang",      ["mo-tai-khoan-ngan-hang", "lien-ket-ngan-hang",
                                  "huy-lien-ket-ngan-hang",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-kienlongbank",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-hdbank",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-namabank",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-eximbank",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-bidv",
                                  "cach-khac-phuc-loi-khi-lien-lien-ket-ngan-hang-saigonbank",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-vib",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-agribank",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-acb",
                                  "cach-khac-phuc-loi-khi-lien-lien-ket-ngan-hang-vcb",
                                  "cach-khac-phuc-loi-khi-lien-ket-ngan-hang-abbank",
                                  "cach-khac-phuc-loi-khi-lien-lien-ket-ngan-hang",
                                  "dang-ky-diem-thuong-ngan-hang"]),
    ("khuyen-mai",              ["hoi-cao-thu-moi-ban", "toro-xep-banh",
                                  "khieu-nai-khuyen-mai",
                                  "thong-tin-chung-ve-khuyen-mai-cua-zalopay"]),
    ("khieu-nai-dich-vu",       ["gop-y-san-pham"]),
    ("zalopay-priority",        ["zalopay-priority"]),
    ("dich-vu-tai-chinh",       ["tai-khoan-tra-sau-lotte-finance", "vay-tien-nhanh",
                                  "tra-gop", "so-du-sinh-loi", "tai-khoan-tra-sau-cimb",
                                  "gui-tiet-kiem", "tai-khoan-chung-khoan", "chung-chi-quy"]),
    ("qr-da-nang",              ["huong-dan-tim-ma-vietqr-trong-ung-dung-ngan-hang",
                                  "nap-tien-bang-qr-da-dang", "nhan-tien-bang-qr-da-nang"]),
    ("dich-vu-thanh-toan-tu-dong", ["apple-service"]),
    ("quet-qr-quoc-te",         ["thanh-toan-quoc-te", "uu-dai-quoc-te"]),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def to_slug(text: str) -> str:
    """Convert Vietnamese text to URL slug."""
    # Normalize and remove diacritics
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("đ", "d").replace("Đ", "d")
    text = text.lower()
    # Replace non-alphanumeric with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text


def discover_article_links(session, category: str, subcategory: str) -> list[str]:
    """
    Fetch a subcategory page and extract article hrefs.
    Returns list of full URLs.
    """
    url = f"{BASE_URL}/hoi-dap/{category}/{subcategory}"
    try:
        r = session.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        pattern = re.compile(rf"^/hoi-dap/{re.escape(category)}/{re.escape(subcategory)}/[^/]+$")
        seen = set()
        links = []
        for a in soup.find_all("a", href=pattern):
            href = a["href"].strip()
            if href not in seen:
                seen.add(href)
                links.append(BASE_URL + href)
        print(f"    discovered {len(links)} links")
        return links
    except Exception as e:
        print(f"    WARN: could not discover links — {e}")
        return []


def fetch_article(session, url: str):
    """
    Fetch an article page and return (title, clean_text).
    Returns (None, None) on failure.
    """
    try:
        r = session.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
    except Exception as e:
        print(f"      ERROR {url}: {e}")
        return None, None

    soup = BeautifulSoup(r.text, "html.parser")

    # ── Title ──
    title = ""
    if (h1 := soup.find("h1")):
        title = h1.get_text(strip=True)
    if not title:
        if (og := soup.find("meta", property="og:title")):
            title = og.get("content", "").strip()
    if not title:
        if (t := soup.find("title")):
            title = t.get_text(strip=True)

    # ── Content: strip nav/header/footer/script/style ──
    for tag in soup.find_all(["nav", "footer", "script", "style", "header",
                               "noscript", "iframe"]):
        tag.decompose()

    content = ""
    # Try known article containers first
    for sel in ["article", '[class*="article-content"]', '[class*="faq-content"]',
                 '[class*="answer"]', "main"]:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 80:
                content = text
                break

    if not content:
        body = soup.find("body")
        if body:
            content = body.get_text(separator="\n", strip=True)

    # Clean up
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    # Remove obvious nav noise
    noise = re.compile(
        r"^(Trang chủ|Trợ giúp|Tin tức|Dịch vụ|Tải Zalopay|HOTLINE|Email:"
        r"|© Copyright|Đối tác|Về ZaloPay|Chăm sóc|Góp ý|Đăng ký doanh nghiệp"
        r"|Báo cáo bảo mật|Hotline|Công ty Cổ phần|Địa chỉ:|Giấy phép|Chứng nhận)$"
    )
    lines = [l for l in lines if not noise.match(l)]
    content = "\n".join(lines[:100])  # keep max 100 lines

    return title or None, content or None


def load_existing_titles(*csv_paths) -> set:
    titles = set()
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row.get("Title", "").strip()
                if t:
                    titles.add(t)
    return titles


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    session = requests.Session()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0530")

    fieldnames = [
        "Canned Response ID", "Title", "Content", "Account ID",
        "Created At", "Updated At", "Content HTML",
        "Folder ID", "Folder Name", "Visibility", "Group Names",
    ]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    main_csv = os.path.join(os.path.dirname(OUTPUT_CSV), "canned_responses.csv")
    existing = load_existing_titles(main_csv, OUTPUT_CSV)
    print(f"Loaded {len(existing)} existing titles to skip duplicates\n")

    new_rows = []
    article_id = 99000000001
    total = 0

    # ── Pass 1: Known articles (Quản lý Tài khoản) ──
    print("=== Pass 1: Known articles (Quản lý Tài khoản) ===")
    for cat, subcat, titles in KNOWN_ARTICLES:
        print(f"\n/{cat}/{subcat}")
        for title in titles:
            slug = to_slug(title)
            url = f"{BASE_URL}/hoi-dap/{cat}/{subcat}/{slug}"
            fetched_title, content = fetch_article(session, url)
            time.sleep(DELAY)

            use_title = fetched_title or title
            if use_title in existing:
                print(f"  SKIP: {use_title[:60]}")
                continue
            if not content:
                print(f"  FAIL: {url}")
                continue

            print(f"  + {use_title[:70]}")
            existing.add(use_title)
            new_rows.append(_row(article_id, use_title, content, now))
            article_id += 1
            total += 1

    # ── Pass 2: Discover other categories ──
    print("\n=== Pass 2: Discover other categories ===")
    for cat, subcats in DISCOVER_CATEGORIES:
        print(f"\n[{cat}]")
        for subcat in subcats:
            print(f"  /{subcat}")
            links = discover_article_links(session, cat, subcat)
            time.sleep(DELAY)

            if not links:
                print(f"    No links found, skipping")
                continue

            for url in links:
                fetched_title, content = fetch_article(session, url)
                time.sleep(DELAY)

                if not fetched_title or not content:
                    continue
                if fetched_title in existing:
                    print(f"    SKIP: {fetched_title[:60]}")
                    continue

                print(f"    + {fetched_title[:70]}")
                existing.add(fetched_title)
                new_rows.append(_row(article_id, fetched_title, content, now))
                article_id += 1
                total += 1

    # ── Write CSV ──
    write_header = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"\n{'='*60}")
    print(f"✅  Scraped {total} new articles  →  {OUTPUT_CSV}")
    print()
    print("Next steps:")
    print("  1. Check the output:  python3 -c \"import csv; r=list(csv.DictReader(open('data/faq_zalopay.csv'))); print(len(r), 'rows')\"")
    print("  2. Merge into main KB: python3 merge_faq.py")
    print("  3. Rebuild Docker:     docker-compose up --build -d")


def _row(aid, title, content, now):
    content_html = "<p>" + content.replace("\n", "</p><p>") + "</p>"
    return {
        "Canned Response ID": aid,
        "Title": title,
        "Content": content,
        "Account ID": 943939,
        "Created At": now,
        "Updated At": now,
        "Content HTML": content_html,
        "Folder ID": 43000108971,
        "Folder Name": "FAQ ZaloPay",
        "Visibility": "Available to all",
        "Group Names": "",
    }


if __name__ == "__main__":
    main()
