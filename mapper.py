# mapper.py
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    # Nếu file có nhiều dấu phẩy, dùng rsplit để lấy đúng label cuối
    parts = line.rsplit(',', 1)
    if len(parts) != 2:
        continue
    img_path, label = parts
    print(f"{img_path}\t{label}")
