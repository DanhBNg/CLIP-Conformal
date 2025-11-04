"""
Compare conformal prediction methods (LAC, APS, RAPS)
Generates a comprehensive comparison report
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_latest_results():
    """Load latest results from MapReduceResult for all methods"""
    base_dir = Path("./MapReduceResult")
    
    results = {}
    
    # Find latest directories for each method
    for method in ['lac', 'aps', 'raps']:
        # Find latest raw data directory
        raw_data_dir = base_dir / "Raw_Data"
        method_dirs = sorted([d for d in raw_data_dir.iterdir() 
                             if d.is_dir() and method in d.name])
        
        if not method_dirs:
            print(f"[SKIP] No results found for {method}")
            continue
        
        latest_dir = method_dirs[-1]
        summary_file = latest_dir / f"summary_{method}.json"
        detailed_file = latest_dir / f"detailed_{method}.json"
        
        if summary_file.exists() and detailed_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            with open(detailed_file, 'r') as f:
                detailed = json.load(f)
            
            results[method] = {
                'summary': summary[0] if isinstance(summary, list) else summary,
                'detailed': detailed,
                'latest_dir': latest_dir
            }
            print(f"[OK] Loaded {method} results from {latest_dir}")
        else:
            print(f"[ERROR] Missing files for {method}")
            print(f"  Looking for: {summary_file} and {detailed_file}")
    
    return results


def create_comparison_table(results):
    """Create comparison table of all methods"""
    
    comparison_data = []
    
    for method, data in results.items():
        summary = data['summary']
        comparison_data.append({
            'Method': method.upper(),
            'Coverage': summary.get('coverage', 'N/A'),
            'Set Size': summary.get('set_size', 'N/A'),
            'Coverage Gap': summary.get('covgap', 'N/A'),
            'Top-1 Acc': summary.get('top1_acc', 'N/A'),
            'Top-5 Acc': summary.get('top5_acc', 'N/A'),
            'Fit Time (ms)': summary.get('time_fit_ms', 'N/A'),
            'Inf Time (ms)': summary.get('time_inf_ms', 'N/A')
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def create_efficiency_metrics(results):
    """Calculate efficiency metrics"""
    
    efficiency = []
    
    for method, data in results.items():
        summary = data['summary']
        coverage = summary.get('coverage', 0)
        set_size = summary.get('set_size', 0)
        fit_time = summary.get('time_fit_ms', 0)
        inf_time = summary.get('time_inf_ms', 0)
        
        # Efficiency score: higher coverage + lower set size + lower inference time
        # Normalize and combine
        if set_size > 0 and inf_time > 0:
            efficiency_score = (coverage * 100) / (set_size * (1 + inf_time/1000))
        else:
            efficiency_score = 0
        
        efficiency.append({
            'Method': method.upper(),
            'Efficiency Score': round(efficiency_score, 4),
            'Coverage (%)': round(coverage * 100, 2),
            'Avg Set Size': round(set_size, 2),
            'Inference Time (ms)': inf_time
        })
    
    df = pd.DataFrame(efficiency)
    df = df.sort_values('Efficiency Score', ascending=False)
    return df


def save_comparison_report(comparison_df, efficiency_df):
    """Save comparison report to Excel"""
    
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_dir = Path("./MapReduceResult/Tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Excel file with multiple sheets
    report_file = output_dir / f"{timestamp}_Conformal_Comparison_Report.xlsx"
    
    with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        efficiency_df.to_excel(writer, sheet_name='Efficiency', index=False)
    
    print(f"\n[OK] Comparison report saved: {report_file}")
    return report_file


def print_detailed_analysis(results):
    """Print detailed analysis and recommendations"""
    
    print("\n" + "="*80)
    print("[ANALYSIS] DETAILED COMPARISON OF CONFORMAL PREDICTION METHODS")
    print("="*80)
    
    for method, data in results.items():
        summary = data['summary']
        detailed = data['detailed']
        
        print(f"\n{method.upper()} Method:")
        print("-" * 60)
        
        # Get dataset results
        dataset_name = list(detailed.keys())[0] if detailed else 'unknown'
        metrics = detailed.get(dataset_name, {})
        
        coverage_values = metrics.get('coverage', [])
        set_size_values = metrics.get('set_size', [])
        
        if coverage_values:
            print(f"  Coverage:          {summary.get('coverage', 0):.4f} (median)")
            print(f"                     Range: [{min(coverage_values):.4f}, {max(coverage_values):.4f}]")
        
        if set_size_values:
            print(f"  Set Size:          {summary.get('set_size', 0):.2f} (median)")
            print(f"                     Range: [{min(set_size_values):.2f}, {max(set_size_values):.2f}]")
        
        print(f"  Top-1 Accuracy:    {summary.get('top1_acc', 0):.4f}")
        print(f"  Fit Time:          {summary.get('time_fit_ms', 0):.2f} ms")
        print(f"  Inference Time:    {summary.get('time_inf_ms', 0):.2f} ms")


def print_recommendations(efficiency_df):
    """Print recommendations based on efficiency"""
    
    print("\n" + "="*80)
    print("[RECOMMENDATIONS]")
    print("="*80)
    
    if len(efficiency_df) > 0:
        best = efficiency_df.iloc[0]
        print(f"\nBest Overall Efficiency: {best['Method']}")
        print(f"  - Efficiency Score: {best['Efficiency Score']}")
        print(f"  - Coverage: {best['Coverage (%)']}%")
        print(f"  - Avg Set Size: {best['Avg Set Size']}")
        print(f"  - Inference Time: {best['Inference Time (ms)']} ms")
    
    # Coverage vs Speed tradeoff
    print("\nCoverage vs Speed Tradeoff:")
    for idx, row in efficiency_df.iterrows():
        print(f"  {row['Method']}: {row['Coverage (%)']}% coverage, "
              f"{row['Inference Time (ms)']} ms inference time")
    
    print("\n" + "="*80)


def main():
    print("\n[START] Loading conformal prediction results...")
    
    # Load results
    results = load_latest_results()
    
    if not results:
        print("[ERROR] No results found!")
        return
    
    # Create comparison table
    print("\n[PROCESS] Creating comparison table...")
    comparison_df = create_comparison_table(results)
    
    # Create efficiency metrics
    print("[PROCESS] Calculating efficiency metrics...")
    efficiency_df = create_efficiency_metrics(results)
    
    # Print detailed analysis
    print_detailed_analysis(results)
    
    # Print efficiency table
    print("\n" + "="*80)
    print("[EFFICIENCY RANKING]")
    print("="*80)
    print(efficiency_df.to_string(index=False))
    
    # Print recommendations
    print_recommendations(efficiency_df)
    
    # Save report
    print("\n[PROCESS] Saving comparison report...")
    save_comparison_report(comparison_df, efficiency_df)
    
    print("\n[OK] Comparison complete!")


if __name__ == "__main__":
    main()
