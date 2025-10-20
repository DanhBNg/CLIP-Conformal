#!/usr/bin/env python3
"""
TEST: Tạo biểu đồ với timing data rõ ràng hơn
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "hadoop" / "MapReduce"))

from draw_charts import create_visualization_charts

def test_charts_with_timing():
    """Test chart generation with clear timing explanation"""
    
    # Sample data giống như kết quả thực tế
    methods_data = {
        0.1: {  # Use float instead of string
            'LAC': {
                'coverage': 0.900,
                'avg_set_size': 8.4,
                'processing_time': 12.5  # seconds
            },
            'APS': {
                'coverage': 0.945,
                'avg_set_size': 35.1,
                'processing_time': 25.0
            },
            'RAPS': {
                'coverage': 0.900,
                'avg_set_size': 9.8,
                'processing_time': 31.3
            }
        },
        '_stage_times': {
            'data_preparation': 2.85,
            'map_phase': 1829.92,      # 30.5 minutes
            'shuffle_sort': 0.28,
            'reduce_phase': 31.31,     # 0.52 minutes
            'total_time': 1864.37      # 31.1 minutes
        }
    }
    
    print("📊 TESTING CHART GENERATION WITH CLEAR TIMING")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("test_charts_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate charts
    session_charts_dir = create_visualization_charts(methods_data, output_dir)
    
    print(f"\n✅ Charts created in: {session_charts_dir}")
    print("\n📋 Generated files:")
    for chart_file in session_charts_dir.glob("*.png"):
        print(f"  📈 {chart_file.name}")
    
    # Print timing explanation
    stage_times = methods_data['_stage_times']
    total_time = stage_times['total_time']
    
    print(f"\n📊 TIMING EXPLANATION:")
    print(f"  Total Pipeline Time: {total_time/60:.1f} minutes")
    print(f"  - Map Phase (CLIP):  {stage_times['map_phase']/60:.1f} min ({(stage_times['map_phase']/total_time)*100:.1f}%)")
    print(f"  - Reduce Phase:      {stage_times['reduce_phase']:.1f} sec ({(stage_times['reduce_phase']/total_time)*100:.1f}%)")
    print(f"  Chart shows algorithm times within Reduce Phase (31.3s total)")
    
    return True

if __name__ == "__main__":
    success = test_charts_with_timing()
    sys.exit(0 if success else 1)