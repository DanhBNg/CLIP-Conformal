# reducer.py
import sys

from collections import defaultdict

label_count = defaultdict(int)

for line in sys.stdin:
    img_path, label = line.strip().split('\t')
    label_count[label] += 1

for label, count in label_count.items():
    print(f"{label}\t{count}")
