#!/usr/bin/env python3
"""
GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
- SUN397 dataset (39,700 ảnh) chia thành 10 chunks (~3,970 ảnh/chunk)
- Load balancing tối ưu cho worker nodes
- Scene recognition với 397 classes
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import subprocess
import math
import torch
import clip

# Canonical project root and canonical hadoop_input directory
project_root = Path(__file__).parent.parent.parent
canonical_hadoop_input = project_root / "hadoop_input"
canonical_hadoop_input.mkdir(parents=True, exist_ok=True)

def load_sun397_dataset():
    """
    Load SUN397 dataset từ thư mục duy nhất (đã merge train + test)
    Returns: all_images (39,700 ảnh), scene_classes (397 classes)
    """
    # Use absolute path từ project root
    project_root = Path(__file__).parent.parent.parent
    sun397_root = project_root / "local_data" / "datasets" / "sun397"
    
    # Check for merged directory first (preferred)
    if (sun397_root / "all").exists():
        data_dir = sun397_root / "all"
        print(f"📁 Loading MERGED SUN397 dataset from {data_dir}")
    elif sun397_root.exists() and any(sun397_root.iterdir()):
        # If no 'all' folder, assume sun397 root contains the classes directly
        data_dir = sun397_root
        print(f"📁 Loading SUN397 dataset from {data_dir}")
    else:
        raise FileNotFoundError(f"SUN397 dataset not found in {sun397_root}")
    
    # Get all scene classes từ thư mục
    scene_classes = []
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            scene_classes.append(class_dir.name)
    
    print(f"🏞️  Found {len(scene_classes)} scene classes")
    
    all_images = []
    
    print(f"📂 Processing all images...")
    for class_idx, scene_name in enumerate(scene_classes):
        class_dir = data_dir / scene_name
        
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} not found")
            continue
            
        # Load tất cả ảnh trong class này
        image_files = list(class_dir.glob("*.jpg"))
        
        for img_path in image_files:
            # Detect original split from filename if available
            split_info = "unknown"
            if img_path.name.startswith("train_"):
                split_info = "train"
            elif img_path.name.startswith("test_"):
                split_info = "test"
            
            image_record = {
                'image_path': str(img_path.absolute()),
                'class_name': scene_name,
                'class_idx': class_idx,
                'split': split_info  # Thông tin split nếu có
            }
            all_images.append(image_record)
    
    print(f"✅ Loaded {len(all_images)} images across {len(scene_classes)} scene classes")
    
    # Count splits if available
    train_count = len([img for img in all_images if img['split'] == 'train'])
    test_count = len([img for img in all_images if img['split'] == 'test'])
    unknown_count = len([img for img in all_images if img['split'] == 'unknown'])
    
    if train_count > 0 or test_count > 0:
        print(f"   📊 Train: {train_count} images, Test: {test_count} images, Unknown: {unknown_count} images")
    
    return all_images, scene_classes

def create_sun397_chunks(all_images, num_chunks=10):
    """
    Chia SUN397 dataset thành chunks
    Args:
        all_images: List of image records
        num_chunks: Số chunks (default: 10)
    """
    total_images = len(all_images)
    chunk_size = math.ceil(total_images / num_chunks)
    
    print(f"✂️  Creating {num_chunks} chunks from {total_images} images")
    print(f"   📦 Chunk size: ~{chunk_size} images each")
    
    # Shuffle để đảm bảo distribution đồng đều
    np.random.shuffle(all_images)
    
    chunks = []
    
    for i in range(0, total_images, chunk_size):
        chunk = all_images[i:i+chunk_size]
        chunks.append(chunk)
        print(f"  Chunk {len(chunks)}: {len(chunk)} images")
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def save_chunks_to_files(chunks):
    """Save chunks to .npz files"""
    print("💾 Saving chunks to files...")
    
    # Tạo thư mục temp
    chunks_dir = Path("temp") / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        # Extract image paths và labels
        image_paths = [record['image_path'] for record in chunk]
        labels = [record['class_idx'] for record in chunk]
        
        # Save as npz file
        chunk_file = chunks_dir / f"chunk_{i+1}.npz"
        np.savez(chunk_file, 
                image_paths=image_paths,
                labels=labels)
        
        chunk_files.append(chunk_file)
        print(f"  📦 Saved chunk {i+1}: {len(image_paths)} images -> {chunk_file}")
    
    print(f"✅ All chunks saved to {chunks_dir}")
    return chunk_files

def create_class_descriptions(scene_classes):
    """Tạo text descriptions cho CLIP text encoder"""
    print("📝 Creating class descriptions for CLIP text encoder...")
    
    # Initialize CLIP for text encoding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Create descriptive prompts cho mỗi scene class
    descriptions = []
    for scene in scene_classes:
        # Create descriptive prompt for scene
        prompt = f"a photo of a {scene.replace('_', ' ')}"
        descriptions.append(prompt)
    
    print(f"  📝 Created descriptions for {len(descriptions)} classes")
    
    # Encode text descriptions
    text_tokens = clip.tokenize(descriptions).to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
    
    # Save encoded features into canonical hadoop_input
    descriptions_dir = canonical_hadoop_input
    descriptions_file = descriptions_dir / "class_descriptions.npy"
    np.save(descriptions_file, text_features.cpu().numpy())
    
    print(f"✅ Class descriptions saved to: {descriptions_file}")
    return descriptions_file

def create_dtd_chunks(all_images, chunk_size=1410):
    """
    LEGACY FUNCTION - No longer used in SUN397 processing.
    Original DTD dataset chunking implementation:
    - DTD có 47 classes
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
    output_dir = canonical_hadoop_input / "dtd_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_files = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_file = output_dir / f"dtd_chunk_{chunk_idx+1:02d}.txt"
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for img_data in chunk:
                line = f"{img_data['image_path']},{img_data['class_idx']},{img_data['class_name']}\n"
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
    
    # Lưu class descriptions to canonical hadoop_input
    desc_file = canonical_hadoop_input / "dtd_class_descriptions.json"
    with open(desc_file, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    print(f"[+] Created class descriptions: {desc_file}")
    return desc_file
    
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
    LEGACY FUNCTION - No longer used. 
    Main function - Chuẩn bị dữ liệu DTD cho MapReduce (OLD)
    """
    print("=== DTD DATASET PREPARATION FOR MAPREDUCE (LEGACY) ===")
    print("=" * 50)
    
    # Bước 1: Load SUN397 dataset
    print("\n[+] Step 1: Loading SUN397 dataset...")
    all_images, scene_classes = load_sun397_dataset()
    
    if len(all_images) == 0:
        print("[-] No SUN397 images found!")
        return False
    
    # Bước 2: Chia thành 10 chunks
    print(f"\n[+] Step 2: Creating 10 chunks...")
    chunks = create_sun397_chunks(all_images)
    
    print(f"[+] Created {len(chunks)} chunks")
    
    # Bước 3: Lưu chunks thành files
    print(f"\n[+] Step 3: Saving chunks to files...")
    chunk_files = save_chunks_to_files(chunks)
    
    # Bước 4: Tạo class descriptions
    print(f"\n[+] Step 4: Creating class descriptions...")
    class_desc_file = create_class_descriptions(scene_classes)
    
    # Bước 5: Upload lên HDFS (optional)
    print(f"\n[+] Step 5: Preparing for processing...")
    print(f"[+] Files ready for MapReduce processing")
    
    # Summary
    print(f"\n[+] SUN397 DATA PREPARATION COMPLETED!")
    print(f"[+] Total images: {len(all_images)}")
    print(f"[+] Scene classes: {len(scene_classes)}")
    print(f"[+] Chunks created: {len(chunks)}")
    print(f"[+] Ready for CLIP-Conformal MapReduce!")
    
    return True
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)