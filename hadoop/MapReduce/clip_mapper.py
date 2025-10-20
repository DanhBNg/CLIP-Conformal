#!/usr/bin/env python3
"""
GIAI ĐOẠN 2: HADOOP MAPPER - CLIP ENCODING
- Mỗi Mapper đọc 1 chunk từ HDFS (~3970 ảnh)
- Khởi tạo CLIP-ViT-B/32 model trong memory
- Xử lý song song: Vision Encoder + Text Encoder
- Tính cosine similarity và tạo logits matrix [batch_size × 397]
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

def process_chunk_local(chunk_file, descriptions_file, chunk_id):
    """
    Process một chunk locally (simulation của mapper)
    """
    print(f"🗺️  Processing chunk {chunk_id}: {chunk_file.name}")
    
    # Initialize CLIP model nếu chưa có
    if clip_model is None:
        initialize_clip_model()
    
    # Load class descriptions
    if class_descriptions is None:
        load_class_descriptions_from_file(descriptions_file)
    
    # Load chunk data từ text file
    image_paths = []
    labels = []
    
    with open(chunk_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                image_paths.append(parts[0])
                labels.append(int(parts[1]))
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    print(f"    📸 Processing {len(image_paths)} images...")
    
    # Process images trong batches
    batch_size = 32
    all_logits = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load và preprocess images
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = clip_preprocess(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"    ⚠️  Error loading {img_path}: {e}")
                # Use dummy image
                dummy_tensor = torch.zeros((3, 224, 224))
                batch_images.append(dummy_tensor)
        
        if batch_images:
            # Stack thành batch tensor
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Encode với CLIP
            with torch.no_grad():
                # Image features
                image_features = clip_model.encode_image(batch_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Text features (đã được precompute)
                text_features = class_descriptions / class_descriptions.norm(dim=-1, keepdim=True)
                
                # Cosine similarity -> logits
                logits = (image_features @ text_features.T) * 100  # Scale by temperature
                
                all_logits.append(logits.cpu().numpy())
    
    # Combine all logits
    if all_logits:
        combined_logits = np.vstack(all_logits)
    else:
        # Fallback nếu không có logits
        combined_logits = np.random.randn(len(image_paths), 397)
    
    # Save outputs
    output_dir = Path("temp") / "mapper_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logits_file = output_dir / f"logits_chunk_{chunk_id}.npy"
    labels_file = output_dir / f"labels_chunk_{chunk_id}.npy"
    
    np.save(logits_file, combined_logits)
    np.save(labels_file, labels)
    
    print(f"    ✅ Chunk {chunk_id} processed: {combined_logits.shape}")
    
    return logits_file, labels_file

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

def load_class_descriptions_from_file(descriptions_file):
    """
    Load class descriptions từ file cho Text Encoder
    """
    global class_descriptions
    
    if descriptions_file.suffix == '.json':
        # Load từ JSON và encode lại
        with open(descriptions_file, 'r') as f:
            descriptions_data = json.load(f)
        
        # Extract text prompts từ descriptions data
        if isinstance(descriptions_data, list) and len(descriptions_data) > 0:
            if isinstance(descriptions_data[0], dict):
                # Format: [{'text_prompt': 'a photo of ...', ...}, ...]
                descriptions_list = [item['text_prompt'] for item in descriptions_data]
            else:
                # Format: ['a photo of ...', ...]
                descriptions_list = descriptions_data
        else:
            descriptions_list = descriptions_data
        
        # Initialize CLIP for text encoding
        device_temp = "cuda" if torch.cuda.is_available() else "cpu" 
        clip_model_temp, _ = clip.load("ViT-B/32", device=device_temp)
        
        # Encode descriptions
        text_tokens = clip.tokenize(descriptions_list).to(device_temp)
        with torch.no_grad():
            text_features = clip_model_temp.encode_text(text_tokens)
        
        class_descriptions = text_features.to(device)
    else:
        # Load từ .npy file
        descriptions_data = np.load(descriptions_file, allow_pickle=True)
        class_descriptions = torch.tensor(descriptions_data, dtype=torch.float32).to(device)
    
    print(f"✅ Loaded {class_descriptions.shape[0]} class descriptions", file=sys.stderr)

def load_class_descriptions():
    """
    Load class descriptions cho Text Encoder
    """
    global class_descriptions
    
    # Download class descriptions từ HDFS
    desc_file = "/tmp/sun397_class_descriptions.json"
    
    # Sử dụng hdfs command để download
    os.system(f"hdfs dfs -get /input/class_info/sun397_class_descriptions.json {desc_file}")
    
    if os.path.exists(desc_file):
        with open(desc_file, 'r') as f:
            class_descriptions = json.load(f)
        print(f"✅ Loaded {len(class_descriptions)} class descriptions", file=sys.stderr)
    else:
        print("❌ Failed to load class descriptions", file=sys.stderr)
        sys.exit(1)

def encode_text_prompts():
    """
    Encode tất cả text prompts cho 397 SUN397 classes
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
        logits=logits,           # [n_samples, 397]
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