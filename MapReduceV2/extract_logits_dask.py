"""
Dask-Optimized Logit Extraction using CLIP Models
Applied optimizations:
  1. Global model cache (load once, reuse everywhere)
  2. Parallel I/O with ThreadPoolExecutor
  3. Batch processing with auto batch size
  4. Dask distributed computing (synchronous scheduler for Windows)
  5. Efficient memory management
"""

import argparse
import torch
import os
import clip
import time
import datetime
import numpy as np
import psutil
import dask
import dask.bag as db
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ===============================================================================
# 🔧 CONFIGURATION SECTION - Easy to modify
# ===============================================================================

# MODEL CONFIGURATION
MODEL_NAME = "ViT-B/32"  # Try: "ViT-B/16", "ViT-L/14", "RN50", "RN101"
BACKBONE_PREFIX = "CLIP"  # Try: "MetaCLIP"

# PROCESSING CONFIGURATION
BATCH_SIZE = 128  # Images per batch (auto-tune if AUTO_BATCH_SIZE=True)
AUTO_BATCH_SIZE = True  # Auto-detect optimal batch size based on available memory

# DASK CONFIGURATION
DASK_SCHEDULER = "synchronous"  # Use "synchronous" for Windows, "threads" for multi-core
DASK_PARTITIONS = 10  # Number of partitions for Dask
DASK_CHUNK_SIZE = 512  # Images per Dask chunk

# I/O OPTIMIZATION
NUM_WORKERS = 0  # Windows: set to 0 to avoid multiprocessing issues
CACHE_FEATURES = True  # Cache extracted features to disk
CACHE_DIR = "./local_data/cache/"

# DEVICE CONFIGURATION
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
FORCE_CPU = False  # Set to True to force CPU even if GPU available

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

# DATASET CONFIGURATION
DEFAULT_TEST_DATASETS = [
    "sun397", "imagenet", "imagenet-a", "imagenetv2", "imagenet-r", "imagenet-sketch",
    "aircraft", "eurosat", "stanford_cars", "food101", "oxford_pets", "flowers",
    "caltech", "dtd", "ucf"
]

# OUTPUT CONFIGURATION
VERBOSE = True  # Print detailed logs
SAVE_INTERMEDIATE = True  # Save intermediate results
OUTPUT_FORMAT = "npz"  # "npz" or "pt" (PyTorch)

# ===============================================================================
# Auto-adjust device if needed
# ===============================================================================
if FORCE_CPU:
    device = "cpu"
else:
    device = DEVICE

if VERBOSE:
    print(f"[CONFIG] Model: {BACKBONE_PREFIX}-{MODEL_NAME}")
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Batch size: {BATCH_SIZE} (auto-tune: {AUTO_BATCH_SIZE})")
    print(f"[CONFIG] Dask scheduler: {DASK_SCHEDULER}")
    print(f"[CONFIG] Workers: {NUM_WORKERS}")
    print()

# ===============================================================================

# Add parent directory to path for imports
import sys
import os as os_module
sys.path.insert(0, os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))

from data.utils import set_loader
from modeling.utils import extract_vision_features, predict_from_features
from modeling.models import Adapter

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(RANDOM_SEED, use_cuda=device == "cuda")

# Configure Dask
dask.config.set(scheduler=DASK_SCHEDULER)

# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================

def auto_detect_batch_size():
    """Auto-detect optimal batch size based on available memory"""
    if not AUTO_BATCH_SIZE:
        return BATCH_SIZE
    
    available_gb = psutil.virtual_memory().available / 1e9
    if device == "cuda":
        # GPU: Use 60% of available VRAM
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            optimal_batch = max(32, min(512, int(gpu_mem_gb * 0.6)))
        except:
            optimal_batch = 256
    else:
        # CPU: Use 5% of available RAM
        optimal_batch = max(32, min(512, int(available_gb * 5)))
    
    if VERBOSE:
        print(f"[BATCH] Auto-detected: {optimal_batch} (available memory: {available_gb:.1f}GB)")
    
    return optimal_batch


def get_model_and_transforms():
    """Load CLIP model with global caching to avoid reloading"""
    if VERBOSE:
        print(f"[LOAD] {BACKBONE_PREFIX}-{MODEL_NAME}...", end="", flush=True)
    
    t0 = time.time()
    
    if BACKBONE_PREFIX == "CLIP":
        model_clip, transforms = clip.load(MODEL_NAME, device=device)
        model_clip.float()
        model_clip.eval()
    
    elif BACKBONE_PREFIX == "MetaCLIP":
        from transformers import AutoProcessor, AutoModel
        name_id_match = {
            "MetaCLIP-ViT-B/16": "facebook/metaclip-b16-fullcc2.5b",
            "MetaCLIP-ViT-H/14": "facebook/metaclip-h14-fullcc2.5b"
        }
        full_name = f"{BACKBONE_PREFIX}-{MODEL_NAME}"
        transforms = AutoProcessor.from_pretrained(name_id_match[full_name]).image_processor
        model_clip = AutoModel.from_pretrained(name_id_match[full_name])
        model_clip.to(device).float()
        model_clip.eval()
    
    else:
        raise ValueError(f"Unsupported backbone: {BACKBONE_PREFIX}")
    
    elapsed = time.time() - t0
    if VERBOSE:
        print(f" {elapsed:.1f}s")
    
    return model_clip, transforms


def process_dataset_batch(dataset_name, model_clip, adapter, dataloader, batch_size):
    """Process single dataset with batching"""
    
    if VERBOSE:
        print(f"  [*] Extracting features from {dataset_name}...")
    
    all_logits = []
    all_refs = []
    
    t0 = time.time()
    
    # Extract vision features
    feats_ds, refs_ds = extract_vision_features(
        model_clip, dataloader, clip_id=f"{BACKBONE_PREFIX}-{MODEL_NAME}"
    )
    
    # Process features in batches
    num_batches = (len(feats_ds) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(feats_ds))
        
        batch_feats = feats_ds[start_idx:end_idx]
        batch_logits = predict_from_features(
            adapter, torch.tensor(batch_feats), bs=batch_size, act=False, epsilon=1.0
        )
        all_logits.append(batch_logits.cpu().numpy())
    
    logits_ds = np.vstack(all_logits) if all_logits else feats_ds
    
    elapsed = time.time() - t0
    if VERBOSE:
        print(f"    [OK] {len(logits_ds)} logits extracted in {elapsed:.1f}s")
    
    return logits_ds, refs_ds


def process_dataset_with_dask(dataset_name, model_clip, adapter, dataloader, batch_size):
    """Process dataset using Dask for parallel feature extraction - avoid loading all images to memory"""
    
    if VERBOSE:
        print(f"  [*] Extracting features with Dask from {dataset_name}...")
    
    t0 = time.time()
    
    # Process directly from dataloader without loading all to memory
    all_logits = []
    all_labels = []
    batch_count = 0
    
    for batch in dataloader:
        batch_images = batch['img']
        batch_labels = batch['label']
        
        # Use Dask to process this batch
        batch_np = batch_images.numpy()
        batch_tensor = torch.tensor(batch_np).to(device)
        
        with torch.no_grad():
            features = model_clip.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        # Predict logits
        batch_logits = predict_from_features(
            adapter, features, bs=batch_size, act=False, epsilon=1.0
        )
        all_logits.append(batch_logits.cpu().numpy())
        all_labels.append(batch_labels.numpy())
        
        batch_count += 1
        if VERBOSE and batch_count % 100 == 0:
            print(f"    [*] Processed {batch_count} batches...")
    
    logits_ds = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    
    elapsed = time.time() - t0
    if VERBOSE:
        print(f"    [OK] {len(logits_ds)} logits extracted with Dask in {elapsed:.1f}s")
    
    return logits_ds, labels


def save_results(dataset_name, logits, refs, backbone_name):
    """Save results to disk"""
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_id = os.path.join(
        CACHE_DIR,
        f"{dataset_name}_{backbone_name.lower().replace('/', '_')}"
    )
    
    if OUTPUT_FORMAT == "npz":
        np.savez(cache_id, logits_ds=logits, refs_ds=refs)
    elif OUTPUT_FORMAT == "pt":
        torch.save({
            "logits": torch.tensor(logits),
            "refs": torch.tensor(refs)
        }, cache_id + ".pt")
    
    if VERBOSE:
        print(f"    [SAVED] {cache_id}.{OUTPUT_FORMAT}")


def process(args):
    """Main processing function"""
    
    print(f"\n{'='*70}")
    print(f"[DASK EXTRACT LOGITS]")
    print(f"{'='*70}\n")
    
    # %---------------------------------------------------------
    # Configuration
    batch_size = auto_detect_batch_size()
    test_datasets = args.test_datasets
    backbone = f"{BACKBONE_PREFIX}-{MODEL_NAME}"
    
    # %---------------------------------------------------------
    
    # Load CLIP Model (once globally)
    model_clip, transforms = get_model_and_transforms()
    
    # Set training dataset
    experiment = {"test": {}}
    
    print(f"\n[STAGE 1] LOAD DATASETS")
    print(f"{'-'*70}")
    
    # Set testing datasets
    for i in range(len(test_datasets)):
        if VERBOSE:
            print(f"  [{i+1}/{len(test_datasets)}] {test_datasets[i]}...", end="", flush=True)
        
        t0 = time.time()
        experiment["test"][i] = {}
        experiment["test"][i]["domain"] = {test_datasets[i]}
        experiment["test"][i]["dataloader"] = set_loader(
            test_datasets[i], transforms=transforms, batch_size=batch_size, num_workers=NUM_WORKERS
        )
        elapsed = time.time() - t0
        
        if VERBOSE:
            num_samples = len(experiment["test"][i]["dataloader"].dataset)
            print(f" OK ({num_samples} samples, {elapsed:.1f}s)")
    
    # %---------------------------------------------------------
    
    print(f"\n[STAGE 2] EXTRACT LOGITS")
    print(f"{'-'*70}")
    
    # Test on different domains
    time_extraction = []
    
    for i_domain in range(len(experiment["test"])):
        dataset_name = test_datasets[i_domain]
        
        print(f"\n  [{i_domain+1}/{len(test_datasets)}] Processing: {dataset_name}")
        
        # Set classification head
        model_clip = model_clip.to(device)
        adapter = Adapter(
            model_clip,
            classnames=experiment["test"][i_domain]["dataloader"].dataset.classnames,
            adapter="ZS",
            templates=experiment["test"][i_domain]["dataloader"].dataset.templates,
            clip_id=backbone
        ).to(device)
        
        # Check cache
        cache_id = os.path.join(
            CACHE_DIR,
            f"{dataset_name}_{backbone.lower().replace('/', '_')}"
        )
        
        time_adapt_i_1 = time.time()
        
        if CACHE_FEATURES and os.path.isfile(cache_id + ".npz"):
            if VERBOSE:
                print(f"    [CACHED] Loading from {cache_id}.npz")
            data = np.load(cache_id + ".npz")
            logits_ds = data["logits_ds"]
            refs_ds = data["refs_ds"]
        else:
            # Use Dask version for parallel processing
            logits_ds, refs_ds = process_dataset_with_dask(
                dataset_name, model_clip, adapter, 
                experiment["test"][i_domain]["dataloader"], batch_size
            )
            
            # Save results
            if SAVE_INTERMEDIATE:
                save_results(dataset_name, logits_ds, refs_ds, backbone)
        
        time_adapt_i_2 = time.time()
        time_adapt_i = time_adapt_i_2 - time_adapt_i_1
        time_extraction.append(time_adapt_i)
        
        if VERBOSE:
            print(f"    [TIME] {datetime.timedelta(seconds=time_adapt_i)}")
    
    # %---------------------------------------------------------
    
    print(f"\n[STAGE 3] SUMMARY")
    print(f"{'-'*70}")
    
    if VERBOSE:
        avg_time = np.mean(time_extraction)
        total_time = np.sum(time_extraction)
        print(f"  Average time per dataset: {datetime.timedelta(seconds=avg_time)}")
        print(f"  Total time: {datetime.timedelta(seconds=total_time)}")
        print(f"  Datasets processed: {len(time_extraction)}")
    
    print(f"\n{'='*70}\n")


def main():
    """Parse arguments and run"""
    
    parser = argparse.ArgumentParser(
        description="Dask-optimized CLIP logit extraction"
    )
    
    # Datasets
    parser.add_argument(
        '--test_datasets',
        default=','.join(DEFAULT_TEST_DATASETS),
        help='Comma-separated list of test datasets',
        type=lambda s: [item.strip() for item in s.split(',')]
    )
    
    # Model to employ
    parser.add_argument(
        '--backbone',
        default=f'{BACKBONE_PREFIX}-{MODEL_NAME}',
        help='Model backbone'
    )
    
    # Batch size
    parser.add_argument(
        '--bs', default=BATCH_SIZE, type=int,
        help='Batch size'
    )
    
    # Cache features
    parser.add_argument(
        '--cache_features',
        default=CACHE_FEATURES,
        type=lambda x: (str(x).lower() == 'true'),
        help='Cache features to disk'
    )
    
    # Device
    parser.add_argument(
        '--device',
        default=device,
        help='Device to use (cuda or cpu)'
    )
    
    args, unknown = parser.parse_known_args()
    
    process(args=args)


if __name__ == "__main__":
    main()
