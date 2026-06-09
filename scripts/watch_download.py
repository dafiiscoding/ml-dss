"""
Theo dõi tiến độ tải ảnh CrisisMMD theo thời gian thực.

Chạy trong terminal của bạn:
    python scripts/watch_download.py
(Nhấn Ctrl+C để dừng theo dõi — KHÔNG ảnh hưởng tới việc tải.)
"""
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = os.path.join(REPO, "data", "raw", "CrisisMMD_v2.0.tar.gz")
TOTAL = 1_896_000_000  # ~1.77 GiB

def fmt(n): return f"{n/1e9:.2f} GB"

def main():
    last_size, last_t = None, None
    print(f"Theo dõi: {FILE}\nTổng ~{fmt(TOTAL)}  (Ctrl+C để thoát)\n")
    while True:
        if not os.path.exists(FILE):
            print("\r[chờ file xuất hiện...]", end="", flush=True); time.sleep(2); continue
        size = os.path.getsize(FILE); now = time.time()
        pct = size / TOTAL * 100
        speed = ""
        if last_size is not None and now > last_t:
            kbps = (size - last_size) / (now - last_t) / 1024
            eta = (TOTAL - size) / ((size - last_size) / (now - last_t) + 1e-9)
            speed = f" | {kbps:6.0f} KB/s | ETA {eta/60:4.1f} phút" if kbps > 1 else " | (đứng yên)"
        bar = "█" * int(pct/2.5) + "░" * (40 - int(pct/2.5))
        print(f"\r{bar} {pct:5.1f}%  {fmt(size)}/{fmt(TOTAL)}{speed}   ", end="", flush=True)
        if size >= TOTAL * 0.999:
            print("\n✔ Tải xong (hoặc gần xong)."); break
        last_size, last_t = size, now
        time.sleep(3)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(Dừng theo dõi — tải vẫn tiếp tục nếu tiến trình tải còn chạy.)")
