#!/usr/bin/env python3
"""
GIAI ĐOẠN 2: HADOOP MAPPER - CLIP ENCODING
- Mỗi Mapper đọc 1 chunk từ HDFS (1410 ảnh)
- Khởi tạo CLIP-ViT-B/32 model trong memory
- Xử lý song song: Vision Encoder + Text Encoder
- Tính cosine similarity và tạo logits matrix [batch_size × 47]
- Output: logits.npz files lưu trở lại HDFS
"""

import sys
import os
import json
import numpy as np
import torch
import clip
from PIL import Image
import tempfile
from pathlib import Path

# Global variables cho CLIP model
clip_model = None
clip_preprocess = None
device = None
class_descriptions = None

def initialize_clip_model():
    """
    Khởi tạo CLIP-ViT-B/32 model trong mapper
    """
    global clip_model, clip_preprocess, device
    
    print("🚀 Initializing CLIP-ViT-B/32 model...", file=sys.stderr)
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}", file=sys.stderr)
    
    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    print("✅ CLIP model loaded successfully", file=sys.stderr)

def load_class_descriptions():
    """
    Load class descriptions cho Text Encoder
    """
    global class_descriptions
    
    # Download class descriptions từ HDFS
    desc_file = "/tmp/dtd_class_descriptions.json"
    
    # Sử dụng hdfs command để download
    os.system(f"hdfs dfs -get /input/class_info/dtd_class_descriptions.json {desc_file}")
    
    if os.path.exists(desc_file):
        with open(desc_file, 'r') as f:
            class_descriptions = json.load(f)
        print(f"✅ Loaded {len(class_descriptions)} class descriptions", file=sys.stderr)
    else:
        print("❌ Failed to load class descriptions", file=sys.stderr)
        sys.exit(1)

def encode_text_prompts():
    """
    Encode tất cả text prompts cho 47 DTD classes
    """
    global clip_model, class_descriptions, device
    
    text_prompts = []
    for desc in class_descriptions:
        text_prompts.append(desc['text_prompt'])
    
    # Tokenize và encode
    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    print(f"✅ Encoded {len(text_prompts)} text prompts", file=sys.stderr)
    return text_features

def process_image_batch(image_paths, class_indices, text_features):
    """
    Xử lý batch ảnh và tính logits
    """
    global clip_model, clip_preprocess, device
    
    batch_logits = []
    batch_labels = []
    batch_image_ids = []
    
    for img_path, class_idx in zip(image_paths, class_indices):
        try:
            # Load và preprocess ảnh
            image = Image.open(img_path).convert('RGB')
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            
            # Vision Encoder
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Tính cosine similarity logits
            logits = (image_features @ text_features.T) * clip_model.logit_scale.exp()
            logits = logits.cpu().numpy().flatten()
            
            batch_logits.append(logits)
            batch_labels.append(class_idx)
            batch_image_ids.append(Path(img_path).stem)
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}", file=sys.stderr)
            continue
    
    return np.array(batch_logits), np.array(batch_labels), batch_image_ids

def save_logits_to_npz(logits, labels, image_ids, chunk_id):
    """
    Lưu logits matrix thành .npz file
    """
    output_dir = Path("/tmp/mapper_output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"logits_chunk_{chunk_id}.npz"
    
    np.savez_compressed(
        output_file,
        logits=logits,           # [n_samples, 47]
        labels=labels,           # [n_samples]
        image_ids=image_ids,     # [n_samples]
        chunk_id=chunk_id
    )
    
    print(f"✅ Saved logits: {output_file} ({logits.shape})", file=sys.stderr)
    return output_file

def upload_results_to_hdfs(output_file):
    """
    Upload kết quả lên HDFS
    """
    hdfs_output_path = f"/output/mapper_logits/{output_file.name}"
    
    # Tạo thư mục output trên HDFS
    os.system("hdfs dfs -mkdir -p /output/mapper_logits")
    
    # Upload file
    result = os.system(f"hdfs dfs -put -f {output_file} {hdfs_output_path}")
    
    if result == 0:
        print(f"✅ Uploaded to HDFS: {hdfs_output_path}", file=sys.stderr)
        return hdfs_output_path
    else:
        print(f"❌ Failed to upload: {hdfs_output_path}", file=sys.stderr)
        return None

def main():
    """
    Main Mapper function
    """
    print("🗂️ CLIP Mapper started", file=sys.stderr)
    
    # Khởi tạo CLIP model
    initialize_clip_model()
    
    # Load class descriptions
    load_class_descriptions()
    
    # Encode text prompts
    text_features = encode_text_prompts()
    
    # Xử lý input từ stdin (chunk file paths)
    image_paths = []
    class_indices = []
    chunk_id = None
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            # Parse: image_path,class_index,class_name,image_id
            parts = line.split(',')
            if len(parts) >= 4:
                img_path = parts[0]
                class_idx = int(parts[1])
                image_id = parts[3]
                
                image_paths.append(img_path)
                class_indices.append(class_idx)
                
                # Extract chunk ID từ image_id
                if chunk_id is None:
                    chunk_id = image_id.split('_')[0] if '_' in image_id else "unknown"
        
        except Exception as e:
            print(f"❌ Error parsing line: {line}, {e}", file=sys.stderr)
            continue
    
    print(f"📊 Processing {len(image_paths)} images in chunk {chunk_id}", file=sys.stderr)
    
    if len(image_paths) == 0:
        print("❌ No valid images to process", file=sys.stderr)
        sys.exit(1)
    
    # Xử lý batch ảnh
    print("🖼️ Encoding images and computing logits...", file=sys.stderr)
    logits, labels, image_ids = process_image_batch(image_paths, class_indices, text_features)
    
    if len(logits) == 0:
        print("❌ No images processed successfully", file=sys.stderr)
        sys.exit(1)
    
    # Lưu kết quả
    print("💾 Saving results...", file=sys.stderr)
    output_file = save_logits_to_npz(logits, labels, image_ids, chunk_id)
    
    # Upload lên HDFS
    hdfs_path = upload_results_to_hdfs(output_file)
    
    if hdfs_path:
        # Output cho Hadoop Streaming (key\tvalue)
        print(f"{chunk_id}\t{hdfs_path}")
        print(f"✅ Mapper completed: {len(logits)} samples processed", file=sys.stderr)
    else:
        print("❌ Mapper failed", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()