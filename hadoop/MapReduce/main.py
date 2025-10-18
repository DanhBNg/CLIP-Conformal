import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import conformal prediction modules
from conformal.conformal_methods import lac, aps, raps
from conformal.metrics import evaluate_conformal, accuracy

# Import chart generation functions
try:
    from draw_charts import create_visualization_charts
except ImportError:
    print("Warning: draw_charts module not found")
    def create_visualization_charts(results, texture_classes):
        print("Charts module not available")

def setup_hadoop_environment():
    """Kiểm tra môi trường Hadoop (không bắt buộc cho standalone mode)"""
    print("🔧 Kiểm tra môi trường...")
    
    # Kiểm tra Hadoop installation (optional)
    try:
        result = subprocess.run(['hadoop', 'version'], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            print("✅ Hadoop có sẵn - có thể chạy distributed mode")
            return True
        else:
            print("📍 Chạy ở chế độ standalone (không cần Hadoop)")
            return False
    except Exception:
        print("📍 Chạy ở chế độ standalone (không cần Hadoop)")
        return False

def stage1_prepare_data():
    """
    GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
    DTD dataset chia thành chunks và upload lên HDFS
    """
    print("\n" + "="*60)
    print("📁 STAGE 1: PREPARING INPUT DATA")
    print("="*60)
    
    # Import prepare_data functions
    from prepare_data import (
        load_dtd_dataset, 
        create_dtd_chunks, 
        save_chunks_to_files,
        create_class_descriptions
    )
    
    # Load DTD dataset (5,640 ảnh)
    all_images, texture_classes = load_dtd_dataset()
    
    # Chia thành 4 chunks (1410 ảnh/chunk) - tương ứng HDFS block size 128MB
    chunks = create_dtd_chunks(all_images, chunk_size=1410)
    
    # Save chunks to files
    chunk_files = save_chunks_to_files(chunks)
    
    # Create class descriptions for text encoder
    descriptions_file = create_class_descriptions(texture_classes)
    
    print("✅ Stage 1 completed: Data prepared for MapReduce")
    return chunk_files, descriptions_file, texture_classes

def stage2_map_phase(chunk_files, descriptions_file):
    """
    GIAI ĐOẠN 2: MAP PHASE - CLIP ENCODING
    Mỗi Mapper xử lý 1 chunk với CLIP-ViT-B/32
    """
    print("\n" + "="*60)
    print("🗺️  STAGE 2: MAP PHASE - CLIP ENCODING")
    print("="*60)
    
    # Import CLIP mapper
    from clip_mapper import process_chunk_local
    
    mapper_outputs = []
    
    # Process mỗi chunk qua mapper (parallel simulation)
    for i, chunk_file in enumerate(chunk_files):
        print(f"\n📝 Processing chunk {i+1}/{len(chunk_files)}: {chunk_file.name}")
        
        # Execute mapper trên chunk này
        logits_file, labels_file = process_chunk_local(chunk_file, descriptions_file, chunk_id=i+1)
        
        mapper_outputs.append({
            'chunk_id': i+1,
            'logits_file': logits_file,
            'labels_file': labels_file
        })
    
    print("✅ Stage 2 completed: All chunks processed by mappers")
    return mapper_outputs

def stage3_shuffle_sort(mapper_outputs):
    """
    GIAI ĐOẠN 3: SHUFFLE & SORT
    Gom nhóm và sắp xếp outputs từ mappers
    """
    print("\n" + "="*60)
    print("🔄 STAGE 3: SHUFFLE & SORT")
    print("="*60)
    
    print("📊 Collecting and sorting mapper outputs...")
    
    # Sort mapper outputs theo chunk_id để đảm bảo consistency
    sorted_outputs = sorted(mapper_outputs, key=lambda x: x['chunk_id'])
    
    # Verify integrity của mapper outputs
    total_samples = 0
    for output in sorted_outputs:
        logits = np.load(output['logits_file'])
        labels = np.load(output['labels_file'])
        total_samples += len(logits)
        print(f"  Chunk {output['chunk_id']}: {len(logits)} samples")
    
    print(f"📈 Total samples collected: {total_samples}")
    print("✅ Stage 3 completed: Data shuffled and sorted")
    return sorted_outputs

def stage4_reduce_phase(sorted_outputs, texture_classes):
    """
    GIAI ĐOẠN 4: REDUCE PHASE - CONFORMAL PREDICTION
    Gộp toàn bộ logits và apply Conformal Prediction
    """
    print("\n" + "="*60)
    print("📉 STAGE 4: REDUCE PHASE - CONFORMAL PREDICTION")
    print("="*60)
    
    # Import conformal prediction modules
    from conformal_reducer import aggregate_mapper_outputs, run_conformal_algorithms
    
    # Aggregate toàn bộ logits từ mappers thành ma trận [total_samples × 47]
    print("🔗 Aggregating logits from all mappers...")
    combined_logits, combined_labels = aggregate_mapper_outputs(sorted_outputs)
    
    print(f"📊 Combined logits shape: {combined_logits.shape}")
    print(f"📊 Combined labels shape: {combined_labels.shape}")
    
    # Chia data: calibration (50%) và test (50%)
    print("✂️  Splitting into calibration and test sets...")
    calib_logits, calib_labels, test_logits, test_labels = split_data(combined_logits, combined_labels)
    
    # Apply Conformal Prediction algorithms (LAC, APS, RAPS)
    print("🎯 Running Conformal Prediction algorithms...")
    results = run_conformal_algorithms(calib_logits, calib_labels, test_logits, test_labels)
    
    print("✅ Stage 4 completed: Conformal Prediction finished")
    return results

def split_data(logits, labels, split_ratio=0.5):
    """Split data into calibration and test sets"""
    n_samples = len(logits)
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * split_ratio)
    
    calib_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    calib_logits = logits[calib_indices]
    calib_labels = labels[calib_indices]
    test_logits = logits[test_indices]
    test_labels = labels[test_indices]
    
    print(f"[+] Calibration set: {len(calib_logits)} samples")
    print(f"[+] Test set: {len(test_logits)} samples")
    
    return calib_logits, calib_labels, test_logits, test_labels

# run_conformal_prediction được thay thế bằng run_conformal_algorithms trong conformal_reducer.py

def save_results(results, output_dir=None):
    """Save results to files and create visualization charts"""
    if output_dir is None:
        # Use the original MapReduceResult directory at root level
        output_dir = Path('../../MapReduceResult')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    charts_dir = output_dir / 'Charts'
    reports_dir = output_dir / 'Reports'
    raw_dir = output_dir / 'Raw_Data'
    
    charts_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    # Create visualization charts first
    session_charts_dir = create_visualization_charts(results, output_dir)
    
    # Save detailed JSON results
    detailed_file = raw_dir / 'conformal_prediction_results.json'
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[+] Detailed results: {detailed_file}")
    
    # Save simplified JSON results
    simple_results = {}
    for method, data in results.items():
        simple_results[method] = {
            'coverage': round(data['coverage_rate'], 3),
            'avg_set_size': round(data['avg_set_size'], 2),
            'runtime': round(data['processing_time'], 4)
        }
    
    simple_file = raw_dir / 'results.json'
    with open(simple_file, 'w') as f:
        json.dump(simple_results, f, indent=2)
    print(f"[+] Simple results: {simple_file}")
    
    # Create Excel report
    try:
        import pandas as pd
        
        # Create a comprehensive DataFrame
        excel_data = []
        for method, data in results.items():
            excel_data.append({
                'Method': method,
                'Coverage_Rate': data['coverage_rate'],
                'Coverage_Percentage': f"{data['coverage_rate']*100:.1f}%",
                'Average_Set_Size': data['avg_set_size'],
                'Runtime_Seconds': data['processing_time'],
                'Runtime_MS': f"{data['processing_time']*1000:.1f}ms",
                'Efficiency_Score': data['coverage_rate'] / data['avg_set_size']
            })
        
        df = pd.DataFrame(excel_data)
        
        # Generate timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = reports_dir / f'DTD_Conformal_Prediction_Results_{timestamp}.xlsx'
        
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results Summary', index=False)
            
            # Add raw results sheet
            raw_df = pd.DataFrame(results).T
            raw_df.to_excel(writer, sheet_name='Raw Results')
            
            # Add analysis sheet
            analysis_data = {
                'Metric': ['Best Coverage', 'Smallest Set Size', 'Fastest Runtime', 'Best Efficiency'],
                'Method': [
                    max(results.keys(), key=lambda x: results[x]['coverage_rate']),
                    min(results.keys(), key=lambda x: results[x]['avg_set_size']),
                    min(results.keys(), key=lambda x: results[x]['processing_time']),
                    max(results.keys(), key=lambda x: results[x]['coverage_rate']/results[x]['avg_set_size'])
                ],
                'Value': [
                    f"{max(results.values(), key=lambda x: x['coverage_rate'])['coverage_rate']*100:.1f}%",
                    f"{min(results.values(), key=lambda x: x['avg_set_size'])['avg_set_size']:.2f}",
                    f"{min(results.values(), key=lambda x: x['processing_time'])['processing_time']*1000:.1f}ms",
                    f"{max(results.values(), key=lambda x: x['coverage_rate']/x['avg_set_size'])['coverage_rate']/max(results.values(), key=lambda x: x['coverage_rate']/x['avg_set_size'])['avg_set_size']:.3f}"
                ]
            }
            
            analysis_df = pd.DataFrame(analysis_data)
            analysis_df.to_excel(writer, sheet_name='Analysis', index=False)
        
        print(f"[+] Excel report: {excel_file}")
        
    except ImportError:
        print("[-] pandas not available for Excel export")
    except Exception as e:
        print(f"[-] Error creating Excel report: {e}")
    
    return output_dir
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax2.bar(methods, efficiency_scores, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Efficiency Score\n(Coverage / Set Size × Time)', fontweight='bold')
    ax2.set_title('(b) Method Efficiency Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, efficiency_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    chart_file = charts_dir / 'runtime_analysis.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'[+] Created: {chart_file}')
    
    # Save detailed JSON results
    detailed_file = raw_dir / 'conformal_prediction_results.json'
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[+] Detailed results: {detailed_file}")
    
    # Save simplified JSON results
    simple_results = {}
    for method, data in results.items():
        simple_results[method] = {
            'coverage': data['coverage_rate'],
            'size': data['avg_set_size'],
            'samples': data['total_samples']
        }
    
    simple_file = raw_dir / 'results.json'
    with open(simple_file, 'w') as f:
        json.dump(simple_results, f, indent=2)
    print(f"[+] Simple results: {simple_file}")
    
    # Try to create Excel report
    try:
        import pandas as pd
        
        # Prepare data for Excel
        excel_data = []
        for method, data in results.items():
            excel_data.append({
                'Method': method,
                'Coverage Rate': f"{data['coverage_rate']:.3f}",
                'Coverage %': f"{data['coverage_rate']*100:.1f}%",
                'Avg Set Size': f"{data['avg_set_size']:.2f}",
                'Total Samples': data['total_samples'],
                'Calibration Size': data['calibration_size'],
                'Test Size': data['test_size'],
                'Processing Time (s)': f"{data['processing_time']:.2f}",
                'Target Coverage': f"{data['target_coverage']:.1%}",
                'Alpha': data['alpha']
            })
        
        df = pd.DataFrame(excel_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_file = reports_dir / f'DTD_Conformal_Prediction_Results_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file) as writer:
            df.to_excel(writer, sheet_name='Results Summary', index=False)
            
            # Add raw data sheet
            raw_df = pd.DataFrame(results).T
            raw_df.to_excel(writer, sheet_name='Raw Data')
        
        print(f"[+] Excel report: {excel_file}")
        
    except ImportError:
        print("[-] Pandas not available - skipping Excel report")
    except Exception as e:
        print(f"[-] Error creating Excel: {e}")
    
    return results

def save_results(results, output_dir):
    """Save results to files and create visualizations"""
    print(f'\n[+] Saving results to {output_dir}...')
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    charts_dir = output_dir / 'Charts'
    reports_dir = output_dir / 'Reports'
    raw_dir = output_dir / 'Raw_Data'
    
    charts_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    
    # Create visualization charts first
    create_visualization_charts(results, output_dir)
    
    # Save detailed JSON results
    detailed_file = raw_dir / 'conformal_prediction_results.json'
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[+] Detailed results: {detailed_file}")
    
    # Save simplified JSON results
    simple_results = {}
    for method, data in results.items():
        simple_results[method] = {
            'coverage': data['coverage_rate'],
            'size': data['avg_set_size'],
            'samples': data['total_samples']
        }
    
    simple_file = raw_dir / 'results.json'
    with open(simple_file, 'w') as f:
        json.dump(simple_results, f, indent=2)
    print(f"[+] Simple results: {simple_file}")
    
    # Try to create Excel report
    try:
        import pandas as pd
        
        # Prepare data for Excel
        excel_data = []
        for method, data in results.items():
            excel_data.append({
                'Method': method,
                'Coverage Rate': f"{data['coverage_rate']:.3f}",
                'Coverage %': f"{data['coverage_rate']*100:.1f}%",
                'Avg Set Size': f"{data['avg_set_size']:.2f}",
                'Total Samples': data['total_samples'],
                'Calibration Size': data['calibration_size'],
                'Test Size': data['test_size'],
                'Processing Time (s)': f"{data['processing_time']:.2f}",
                'Target Coverage': f"{data['target_coverage']:.1%}",
                'Alpha': data['alpha']
            })
        
        df = pd.DataFrame(excel_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_file = reports_dir / f'DTD_Conformal_Prediction_Results_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file) as writer:
            df.to_excel(writer, sheet_name='Results Summary', index=False)
            
            # Add raw data sheet
            raw_df = pd.DataFrame(results).T
            raw_df.to_excel(writer, sheet_name='Raw Data')
        
        print(f"[+] Excel report: {excel_file}")
        
    except ImportError:
        print("[-] Pandas not available - skipping Excel report")
    except Exception as e:
        print(f"[-] Error creating Excel: {e}")
    
    return results

def mapreduce_orchestrator():
    """
    MAIN MAPREDUCE ORCHESTRATOR
    Điều phối toàn bộ pipeline MapReduce cho CLIP-Conformal
    Thu thập thời gian thực tế từ mỗi giai đoạn
    """
    print("🚀 STARTING CLIP-CONFORMAL MAPREDUCE PIPELINE")
    print("=" * 80)
    
    total_start_time = time.time()
    
    # Initialize timing data để track thời gian thực
    stage_times = {}
    
    # Setup Hadoop environment
    hadoop_available = setup_hadoop_environment()
    
    try:
        # STAGE 1: Prepare data
        stage_start = time.time()
        chunk_files, descriptions_file, texture_classes = stage1_prepare_data()
        stage_times['data_preparation'] = time.time() - stage_start
        
        # STAGE 2: Map phase - CLIP encoding
        stage_start = time.time()
        mapper_outputs = stage2_map_phase(chunk_files, descriptions_file)
        stage_times['map_phase'] = time.time() - stage_start
        
        # STAGE 3: Shuffle & Sort
        stage_start = time.time()
        sorted_outputs = stage3_shuffle_sort(mapper_outputs)
        stage_times['shuffle_sort'] = time.time() - stage_start
        
        # STAGE 4: Reduce phase - Conformal Prediction
        stage_start = time.time()
        results = stage4_reduce_phase(sorted_outputs, texture_classes)
        stage_times['reduce_phase'] = time.time() - stage_start
        
        # Calculate total time
        total_time = time.time() - total_start_time
        stage_times['total_time'] = total_time
        
        # Add timing data to results để pass cho chart generation
        results['_stage_times'] = stage_times
        
        print("\n" + "="*80)
        print("🎉 MAPREDUCE PIPELINE COMPLETED SUCCESSFULLY")
        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"⏱️  Data prep: {stage_times['data_preparation']:.3f}s")
        print(f"⏱️  Map phase: {stage_times['map_phase']:.3f}s") 
        print(f"⏱️  Shuffle/Sort: {stage_times['shuffle_sort']:.3f}s")
        print(f"⏱️  Reduce phase: {stage_times['reduce_phase']:.3f}s")
        print("="*80)
        
        return results, texture_classes
        
    except Exception as e:
        print(f"\n❌ MAPREDUCE PIPELINE FAILED: {e}")
        raise e

def create_results_summary(results, texture_classes):
    """Create results summary for charts với real timing data"""
    print("\n📊 CREATING RESULTS SUMMARY")
    print("-" * 40)
    
    # Extract timing data if available
    stage_times = results.pop('_stage_times', None)
    
    # Create summary từ results  
    summary = {}
    
    for alpha in results:
        summary[alpha] = {}
        for method in results[alpha]:
            result = results[alpha][method]
            summary[alpha][method] = {
                'coverage': result['coverage'],
                'avg_set_size': result['avg_set_size'],
                'prediction_sets': result['prediction_sets']
            }
    
    # Add timing data back for chart generation
    if stage_times:
        summary['_stage_times'] = stage_times
        print(f"✅ Real timing data added: {stage_times}")
    
    print("✅ Results summary created")
    return summary

def main():
    """Main function chạy MapReduce pipeline thay vì load cache"""
    print('=' * 60)
    print('🎯 CLIP-Conformal Prediction via MapReduce')
    print('=' * 60)
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run MapReduce pipeline thay vì load_dtd_data()
        results, texture_classes = mapreduce_orchestrator()
        
        # Save results to original MapReduceResult directory
        output_dir = Path('../../MapReduceResult')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Raw_Data subdirectory for JSON results
        raw_data_dir = output_dir / 'Raw_Data'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON in Raw_Data folder
        results_file = raw_data_dir / f'conformal_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Convert results to serializable format (skip timing data)
        serializable_results = {}
        for alpha, methods in results.items():
            if alpha == '_stage_times':  # Skip timing data for JSON serialization
                continue
            serializable_results[alpha] = {}
            for method, data in methods.items():
                serializable_results[alpha][method] = {}
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[alpha][method][key] = value.tolist()
                    elif hasattr(value, 'item'):  # scalar numpy types
                        serializable_results[alpha][method][key] = value.item()
                    else:
                        serializable_results[alpha][method][key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Results saved to: {results_file}")
        
        # Create results summary for charts
        summary = create_results_summary(results, texture_classes)
        
        # Generate visualization charts using real results
        print("\n📊 GENERATING VISUALIZATION CHARTS")
        print("-" * 40)
        create_visualization_charts(summary, texture_classes)
        
        # Print summary
        print(f'\n{"="*60}')
        print('📊 CONFORMAL PREDICTION RESULTS SUMMARY:')
        print('='*60)
        
        # Display results for first alpha value
        first_alpha = list(results.keys())[0]
        for method, data in results[first_alpha].items():
            coverage = data['coverage']
            size = data['avg_set_size']
            target = 1 - first_alpha
            
            status = "✓" if coverage >= target * 0.95 else "⚠"
            
            print(f"  {status} {method:4s}: Coverage={coverage:.3f} ({coverage*100:.1f}%), "
                  f"Size={size:.1f}")
        
        print(f'\n[+] Results saved in: {output_dir}')
        print('[+] DTD Conformal Prediction completed successfully via MapReduce!')
        
        return True
        
    except Exception as e:
        print(f'\n❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
