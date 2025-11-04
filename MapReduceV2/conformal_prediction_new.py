"""
Conformal Prediction Pipeline - Su dung logits tu extract_logits_dask.py
Chay conformal prediction algorithms (LAC, APS, RAPS) tren du lieu da extract
"""

import argparse
import torch
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import conformal prediction modules
from conformal.conformal_methods import lac, aps, raps
from conformal.metrics import evaluate_conformal, accuracy
from conformal.split import split_data

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')

print(f"[CONFIG] Device: {device}")


def load_logits_from_cache(dataset_name, backbone_name):
    """Load extracted logits from cache file"""
    cache_file = f"./local_data/cache/{dataset_name}_{backbone_name.lower().replace('/', '_')}.npz"
    
    if not os.path.isfile(cache_file):
        print(f"[ERROR] Cache file not found: {cache_file}")
        return None, None
    
    print(f"[LOAD] Loading logits from: {cache_file}")
    try:
        cache = np.load(cache_file, allow_pickle=True)
        logits_ds = torch.tensor(cache["logits_ds"], dtype=torch.float32)
        labels_ds = torch.tensor(cache["refs_ds"], dtype=torch.long)
        print(f"[OK] Loaded logits: {logits_ds.shape}, labels: {labels_ds.shape}")
        return logits_ds, labels_ds
    except Exception as e:
        print(f"[ERROR] Error loading cache: {e}")
        return None, None


def process_conformal_prediction(args):
    """Run conformal prediction on extracted logits"""
    
    print("\n" + "="*70)
    print("[STAGE] CONFORMAL PREDICTION ANALYSIS")
    print("="*70)
    
    # Prepare results storage
    res = pd.DataFrame()
    results_detailed = {}
    
    # Process each dataset
    for i_domain in range(len(args.test_datasets)):
        dataset_name = args.test_datasets[i_domain]
        print(f"\n[DATASET {i_domain+1}/{len(args.test_datasets)}]: {dataset_name}")
        
        # Load logits from cache
        logits_ds, labels_ds = load_logits_from_cache(dataset_name, args.backbone)
        if logits_ds is None:
            print(f"[SKIP] Skipping {dataset_name}")
            continue
        
        # Normalize labels to 0-indexed
        labels_ds = labels_ds - 1  # Convert from 1-indexed to 0-indexed
        
        print(f"\n[INFO] Data Info")
        print(f"  - Total samples: {len(logits_ds)}")
        print(f"  - Number of classes: {logits_ds.shape[1]}")
        print(f"  - Labels unique: {len(torch.unique(labels_ds))}")
        
        # Initialize metrics collectors
        emp_cov, set_size, class_covgap = [], [], []
        top1, top5 = [], []
        time_adapt, time_conf_fit, time_conf_inf = [], [], []
        
        # Run for multiple seeds
        print(f"\n[EXEC] Running conformal prediction ({args.seeds} seeds)...")
        for seed_idx in tqdm(range(args.seeds), desc="Seeds", leave=True):
            torch.cuda.empty_cache()
            
            # Split data into calibration and test
            logits_calib, labels_calib, logits_test, labels_test = split_data(
                logits_ds, labels_ds, p=args.p
            )
            
            print(f"\n  Seed {seed_idx+1}/{args.seeds}:")
            print(f"    - Calibration: {len(labels_calib)} samples")
            print(f"    - Test: {len(labels_test)} samples")
            
            # Combine both sets for adaptation
            logits_combined = torch.cat([logits_calib, logits_test])
            labels_combined = torch.cat([labels_calib, labels_test])
            
            # Apply temperature scaling or adaptation
            time_adapt_i_1 = time.time()
            
            if args.adapt == "none":
                # No adaptation - temperature scaling softmax of logit scores
                z = torch.softmax(logits_combined / args.epsilon, dim=-1)
            else:
                # For now, use simple temperature scaling
                z = torch.softmax(logits_combined / args.epsilon, dim=-1)
            
            time_adapt_i_2 = time.time()
            time_adapt_i = time_adapt_i_2 - time_adapt_i_1
            
            # Get predictions for calibration and test
            preds_calib = z[:len(labels_calib), :]
            preds_test = z[len(labels_calib):, :]
            
            # Apply conformal prediction algorithm
            time_conf_fit_1 = time.time()
            
            if args.ncscore == 'lac':
                val_sets, time_fit_i, time_infer_i = lac(
                    preds_calib, labels_calib, preds_test, args.alpha
                )
            elif args.ncscore == 'aps':
                val_sets, time_fit_i, time_infer_i = aps(
                    preds_calib, labels_calib, preds_test, args.alpha
                )
            elif args.ncscore == 'raps':
                val_sets, time_fit_i, time_infer_i = raps(
                    preds_calib, labels_calib, preds_test, args.alpha,
                    lambda_raps=0.0001, k_raps=5
                )
            else:
                raise ValueError(f"Unknown ncscore: {args.ncscore}")
            
            time_conf_fit_total = time.time() - time_conf_fit_1
            
            # Evaluate metrics
            metrics_conformal = evaluate_conformal(val_sets, labels_test, alpha=args.alpha)
            metrics_accuracy = accuracy(preds_test, labels_test, (1, 5))
            
            # Store metrics
            emp_cov.append(metrics_conformal[0])
            set_size.append(metrics_conformal[1])
            class_covgap.append(metrics_conformal[2])
            top1.append(metrics_accuracy[0].item())
            top5.append(metrics_accuracy[1].item())
            time_adapt.append(time_adapt_i)
            time_conf_fit.append(time_fit_i)
            time_conf_inf.append(time_infer_i)
            
            # Print intermediate results
            print(f"    Coverage: {np.round(emp_cov[-1], 3)} | Set Size: {np.round(set_size[-1], 2)} | "
                  f"Top-1 Acc: {np.round(top1[-1], 3)}")
        
        # Print summary for dataset
        print(f"\n{'='*70}")
        print(f"[SUMMARY] {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"  Empirical Coverage: {np.round(np.median(emp_cov), 3)}")
        print(f"  Average Set Size:   {np.round(np.median(set_size), 2)}")
        print(f"  Class Coverage Gap: {np.round(np.median(class_covgap), 3)}")
        print(f"  Top-1 Accuracy:     {np.round(np.median(top1), 3)}")
        print(f"  Top-5 Accuracy:     {np.round(np.median(top5), 3)}")
        print(f"  Adapt Time (ms):    {np.round(np.mean(time_adapt)*1000, 2)}")
        print(f"  Fit Time (ms):      {np.round(np.mean(time_conf_fit)*1000, 2)}")
        print(f"  Inference Time (ms):{np.round(np.mean(time_conf_inf)*1000, 2)}")
        print(f"{'='*70}\n")
        
        # Store detailed results
        results_detailed[dataset_name] = {
            "coverage": emp_cov,
            "set_size": set_size,
            "class_covgap": class_covgap,
            "top1": top1,
            "top5": top5,
            "time_adapt": time_adapt,
            "time_conf_fit": time_conf_fit,
            "time_conf_inf": time_conf_inf
        }
        
        # Prepare row for summary table
        res_i = {
            "backbone": args.backbone,
            "dataset": dataset_name,
            "alpha": args.alpha,
            "adapt": args.adapt,
            "ncscore": args.ncscore,
            "epsilon": args.epsilon,
            "prop_calib": args.p,
            "seeds": args.seeds,
            "coverage": np.round(np.median(emp_cov), 3),
            "set_size": np.round(np.median(set_size), 2),
            "covgap": np.round(np.median(class_covgap), 3),
            "top1_acc": np.round(np.median(top1), 3),
            "top5_acc": np.round(np.median(top5), 3),
            "time_adapt_ms": np.round(np.mean(time_adapt)*1000, 2),
            "time_fit_ms": np.round(np.mean(time_conf_fit)*1000, 2),
            "time_inf_ms": np.round(np.mean(time_conf_inf)*1000, 2)
        }
        res = pd.concat([res, pd.DataFrame(res_i, index=[0])], ignore_index=True)
    
    # Save results
    return res, results_detailed


def save_results(res, results_detailed, args):
    """Save results to files"""
    
    print("\n[SAVE] Saving results...")
    
    # Create output directories in MapReduceResult
    base_dir = Path("./MapReduceResult")
    tables_dir = base_dir / "Tables"
    raw_dir = base_dir / "Raw_Data"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    # Create subdirectories with timestamp
    method_tables_dir = tables_dir / f"{timestamp}_{args.ncscore}_tables"
    method_raw_dir = raw_dir / f"{timestamp}_{args.ncscore}_raw"
    
    method_tables_dir.mkdir(exist_ok=True)
    method_raw_dir.mkdir(exist_ok=True)
    
    # Save summary results as Excel
    summary_file = method_tables_dir / f"conformal_{args.ncscore}_summary.xlsx"
    res.to_excel(summary_file, index=False)
    print(f"[OK] Summary saved: {summary_file}")
    
    # Save detailed results as JSON
    detailed_file = method_raw_dir / f"detailed_{args.ncscore}.json"
    # Convert numpy arrays to lists for JSON serialization
    results_for_json = {}
    for dataset, metrics in results_detailed.items():
        results_for_json[dataset] = {}
        for key, values in metrics.items():
            if isinstance(values, list):
                results_for_json[dataset][key] = [float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                                                   for v in values]
    
    with open(detailed_file, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"[OK] Detailed results saved: {detailed_file}")
    
    # Save summary as JSON
    summary_json_file = method_raw_dir / f"summary_{args.ncscore}.json"
    summary_dict = res.to_dict(orient='records')
    with open(summary_json_file, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"[OK] Summary JSON saved: {summary_json_file}")
    
    # Print summary table
    print("\n" + "="*120)
    print("[RESULTS] FINAL SUMMARY TABLE")
    print("="*120)
    print(res.to_string(index=False))
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description="Conformal Prediction on cached CLIP logits")
    
    # Datasets
    parser.add_argument(
        '--test_datasets',
        default='sun397',
        help='Comma-separated list of test datasets',
        type=lambda s: [item.strip() for item in s.split(',')]
    )
    
    # Model backbone
    parser.add_argument(
        '--backbone',
        default='CLIP-ViT-B/32',
        help='Model backbone (e.g., CLIP-ViT-B/32, CLIP-ViT-L/14)'
    )
    
    # Conformal prediction settings
    parser.add_argument('--alpha', default=0.1, type=float, help='Significance level (1-coverage)')
    parser.add_argument('--ncscore', default='lac', choices=['lac', 'aps', 'raps'], 
                        help='Non-conformity score')
    parser.add_argument('--adapt', default='none', choices=['none', 'linear_probe', 'confot'], 
                        help='Adaptation method')
    parser.add_argument('--epsilon', default=1.0, type=float, help='Temperature scaling parameter')
    
    # Data settings
    parser.add_argument('--p', default=0.5, type=float, help='Proportion of calibration data')
    parser.add_argument('--seeds', default=20, type=int, help='Number of random seeds')
    
    args, unknown = parser.parse_known_args()
    
    print(f"\n[CONFIG] Backbone: {args.backbone}")
    print(f"[CONFIG] Alpha (significance): {args.alpha}")
    print(f"[CONFIG] Non-conformity score: {args.ncscore}")
    print(f"[CONFIG] Calibration proportion: {args.p}")
    print(f"[CONFIG] Number of seeds: {args.seeds}")
    print(f"[CONFIG] Datasets: {', '.join(args.test_datasets)}\n")
    
    # Run conformal prediction
    res, results_detailed = process_conformal_prediction(args)
    
    # Save results
    save_results(res, results_detailed, args)
    
    print("\n[OK] Conformal prediction completed!")


if __name__ == "__main__":
    main()
