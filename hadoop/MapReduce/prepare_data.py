#!/usr/bin/env python3
"""
GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
- DTD dataset (5,640 ảnh) chia thành 4 chunks (1410 ảnh/chunk)
- Tương ứng với HDFS block size (128MB)
- Load balancing tối ưu cho worker nodes
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import subprocess
import math

def load_dtd_dataset():
    """Load toàn bộ DTD dataset từ thư mục images"""
    dtd_root = Path("local_data/datasets/dtd/images")
    
    # DTD có 47 texture classes
    texture_classes = [
        'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered',
        'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted',
        'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid',
        'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike',
        'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated',
        'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed',
        'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained',
        'stratified', 'striped', 'studded', 'swirly', 'veined',
        'waffled', 'woven', 'wrinkled', 'zigzagged'
    ]
    
    all_images = []
    
    print(f"[+] Loading DTD dataset from {dtd_root}")
    
    for class_idx, texture_name in enumerate(texture_classes):
        class_dir = dtd_root / texture_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
            
        # Load tất cả ảnh trong class này
        image_files = list(class_dir.glob("*.jpg"))
        
        for img_path in image_files:
            image_record = {
                'image_path': str(img_path.absolute()),
                'class_index': class_idx,
                'class_name': texture_name,
                'image_id': f"dtd_{class_idx:02d}_{img_path.stem}"
            }
            all_images.append(image_record)
        
        print(f"  {texture_name}: {len(image_files)} anh")
    
    print(f"[+] Total DTD images loaded: {len(all_images)}")
    return all_images, texture_classes

def create_dtd_chunks(all_images, chunk_size=1410):
    """
    Chia DTD dataset thành chunks theo yêu cầu:
    - 5,640 ảnh chia thành 4 chunks
    - Mỗi chunk = 1410 ảnh
    - Đảm bảo load balancing
    """
    print(f"[+] Creating chunks with size = {chunk_size}")
    
    # Shuffle để đảm bảo mỗi chunk có đủ các class
    import random
    random.seed(42)  # Reproducible
    random.shuffle(all_images)
    
    chunks = []
    total_images = len(all_images)
    
    for i in range(0, total_images, chunk_size):
        chunk = all_images[i:i + chunk_size]
        chunks.append(chunk)
        
        # Tính class distribution trong chunk
        class_counts = {}
        for img in chunk:
            class_name = img['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"  Chunk {len(chunks)}: {len(chunk)} anh, {len(class_counts)} classes")
    
    return chunks

def save_chunks_to_files(chunks):
    """
    Lưu mỗi chunk thành file .txt để upload lên HDFS
    Format: image_path,class_index,class_name,image_id
    """
    output_dir = Path("hadoop_input/dtd_chunks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_files = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_file = output_dir / f"dtd_chunk_{chunk_idx+1:02d}.txt"
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for img_data in chunk:
                line = f"{img_data['image_path']},{img_data['class_index']},{img_data['class_name']},{img_data['image_id']}\n"
                f.write(line)
        
        # Kiểm tra file size
        file_size_mb = chunk_file.stat().st_size / (1024 * 1024)
        
        chunk_files.append(chunk_file)
        print(f"[+] {chunk_file.name}: {len(chunk)} anh, {file_size_mb:.2f} MB")
    
    return chunk_files

def create_class_descriptions(texture_classes):
    """
    Tạo text descriptions cho CLIP Text Encoder
    """
    descriptions = []
    
    for class_idx, class_name in enumerate(texture_classes):
        desc = {
            'class_index': class_idx,
            'class_name': class_name,
            'text_prompt': f"a photo of {class_name} texture",
            'alternative_prompts': [
                f"{class_name} surface texture",
                f"texture pattern of {class_name}",
                f"{class_name} textured material"
            ]
        }
        descriptions.append(desc)
    
    # Lưu class descriptions
    desc_file = Path("hadoop_input/dtd_class_descriptions.json")
    with open(desc_file, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    
    print(f"[+] Created class descriptions: {desc_file}")
    return desc_file

def upload_to_hdfs(chunk_files, class_desc_file):
    """
    Upload tất cả chunks và class descriptions lên HDFS
    """
    hadoop_bin = r"C:\hadoop-3.3.0\bin\hdfs.cmd"
    
    print(f"[+] Uploading to HDFS...")
    
    # Tạo thư mục HDFS
    subprocess.run([
        hadoop_bin, "dfs", "-mkdir", "-p", "/input/dtd_chunks"
    ], check=False)
    
    subprocess.run([
        hadoop_bin, "dfs", "-mkdir", "-p", "/input/class_info"
    ], check=False)
    
    # Upload chunks
    uploaded_chunks = []
    for chunk_file in chunk_files:
        hdfs_path = f"/input/dtd_chunks/{chunk_file.name}"
        
        result = subprocess.run([
            hadoop_bin, "dfs", "-put", "-f", str(chunk_file), hdfs_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            uploaded_chunks.append(hdfs_path)
            print(f"  [+] {chunk_file.name} -> {hdfs_path}")
        else:
            print(f"  [-] Failed: {chunk_file.name}")
    
    # Upload class descriptions
    hdfs_desc_path = "/input/class_info/dtd_class_descriptions.json"
    result = subprocess.run([
        hadoop_bin, "dfs", "-put", "-f", str(class_desc_file), hdfs_desc_path
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  [+] Class descriptions -> {hdfs_desc_path}")
    
    # Hiển thị HDFS structure
    print(f"\n[+] HDFS Input Structure:")
    subprocess.run([hadoop_bin, "dfs", "-ls", "-h", "/input/dtd_chunks/"])
    subprocess.run([hadoop_bin, "dfs", "-ls", "-h", "/input/class_info/"])
    
    return uploaded_chunks

def main():
    """
    Main function - Chuẩn bị dữ liệu DTD cho MapReduce
    """
    print("=== DTD DATASET PREPARATION FOR MAPREDUCE ===")
    print("=" * 50)
    
    # Bước 1: Load DTD dataset
    print("\n[+] Step 1: Loading DTD dataset...")
    all_images, texture_classes = load_dtd_dataset()
    
    if len(all_images) == 0:
        print("[-] No DTD images found!")
        return False
    
    # Bước 2: Chia thành chunks
    print(f"\n[+] Step 2: Creating chunks...")
    target_chunks = 4
    chunk_size = math.ceil(len(all_images) / target_chunks)
    chunks = create_dtd_chunks(all_images, chunk_size)
    
    print(f"[+] Created {len(chunks)} chunks, ~{chunk_size} anh/chunk")
    
    # Bước 3: Lưu chunks thành files
    print(f"\n[+] Step 3: Saving chunks to files...")
    chunk_files = save_chunks_to_files(chunks)
    
    # Bước 4: Tạo class descriptions
    print(f"\n[+] Step 4: Creating class descriptions...")
    class_desc_file = create_class_descriptions(texture_classes)
    
    # Bước 5: Upload lên HDFS
    print(f"\n[+] Step 5: Uploading to HDFS...")
    uploaded_chunks = upload_to_hdfs(chunk_files, class_desc_file)
    
    # Summary
    print(f"\n[+] DTD DATA PREPARATION COMPLETED!")
    print(f"[+] Total images: {len(all_images)}")
    print(f"[+] Chunks created: {len(chunks)}")
    print(f"[+] HDFS chunks: {len(uploaded_chunks)}")
    print(f"[+] Ready for CLIP-Conformal MapReduce!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)