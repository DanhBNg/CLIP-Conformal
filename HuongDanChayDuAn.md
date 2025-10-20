# 🎯 Hướng Dẫn Chạy Dự Án CLIP-Conformal Prediction

## 📋 Mục Đích Dự Án
Dự án này sử dụng thuật toán **Conformal Prediction** để dự đoán texture (kết cấu bề mặt) của hình ảnh với độ tin cậy cao. Thay vì chỉ đưa ra 1 kết quả dự đoán, hệ thống sẽ đưa ra một **tập kết quả có thể** với mức độ tin cậy 90% hoặc 95%.

**Ví dụ đơn giản**: 
- Dự đoán thông thường: "Đây là texture gỗ" 
- Conformal Prediction: "Đây có thể là texture gỗ hoặc vải, với độ tin cậy 90%"
---

## 📁 Cấu Trúc Dự Án MapReduce

### 🗂️ Thư Mục Chính

```
CLIP-Conformal/
├── 📁 hadoop/mapreduce/         # Core MapReduce Pipeline
│   ├── 📄 main.py              # FILE CHÍNH - MapReduce Orchestrator
│   ├── 📄 prepare_data.py      # Stage 1: Data Preparation & Chunking
│   ├── 📄 clip_mapper.py       # Stage 2: Map Phase - CLIP Encoding
│   ├── � conformal_reducer.py # Stage 4: Reduce Phase - Conformal Algorithms
│   └── 📄 draw_charts.py       # Visualization Engine
├── 📁 hadoop_input/            # MapReduce Input Data (auto-generated)
│   ├── 📄 dtd_chunk_01.txt     # Data chunk 1 (1410 images)
│   ├── � dtd_chunk_02.txt     # Data chunk 2 (1410 images)
│   ├── 📄 dtd_chunk_03.txt     # Data chunk 3 (1410 images)
│   ├── 📄 dtd_chunk_04.txt     # Data chunk 4 (1410 images)
│   └── 📄 dtd_class_descriptions.json # Texture class descriptions
├── 📁 local_data/datasets/dtd/ # Original DTD Dataset
│   ├── 📁 images/              # 5,640 texture images (47 classes)
│   └── 📁 labels/              # Image labels and splits
├── 📁 conformal/               # Conformal Prediction Algorithms
│   ├── � conformal_methods.py # LAC, APS, RAPS implementations
│   ├── 📄 metrics.py           # Evaluation metrics
│   └── 📄 split.py             # Data splitting utilities
├── 📁 MapReduceResult/         # FINAL OUTPUT (auto-generated)
│   ├── 📁 Charts/              # Timestamped visualization folders
│   │   └── 📁 [HHhmm_DD-MM-YYYY_charts]/ # Session-specific charts (7 charts)
│   ├── 📁 Reports/             # 🆕 Excel reports
│   │   └── 📄 [HHhmm_YYYYMMDD.xlsx]      # Excel export với 4 sheets
│   └── 📁 Raw_Data/            # JSON results storage
│       └── 📄 conformal_results_*.json   # Detailed results (dual-alpha)
└── 📄 HuongDanChayDuAn.md     # This guide
```

### 📄 Luồng Xử Lý MapReduce - File Mapping

| Stage | File | Input | Output | Chức năng |
|-------|------|-------|--------|-----------|
| **Stage 1** | `prepare_data.py` | DTD images (5,640) | Chunks (4×1,410) | Data chunking & preparation |
| **Stage 2** | `clip_mapper.py` | Data chunks | CLIP logits | Parallel CLIP-ViT-B/32 encoding |
| **Stage 3** | `main.py` | Mapper outputs | Sorted data | Shuffle & sort aggregation |
| **Stage 4** | `conformal_reducer.py` | Aggregated data | CP results | LAC/APS/RAPS algorithms (dual-alpha) |
| **Visualization** | `draw_charts.py` | CP results | Charts | Real-time visualization (7 charts) |
| **🆕 Export** | `main.py` | CP results | Excel | Research-ready Excel export |

---

## 🚀 Hướng Dẫn Chạy Từng Bước

### Bước 1: Chuẩn Bị Môi Trường
```bash
# Mở terminal/PowerShell tại thư mục dự án
cd C:\BigData\CLIP-Conformal

# Kích hoạt môi trường Python (nếu có)
.venv\Scripts\Activate.ps1  # Windows PowerShell
# hoặc
.venv/Scripts/activate      # Windows CMD
```

### Bước 2: Chạy Dự Án
```bash
# Chạy file chính - CHỈ CẦN LỆNH NÀY!
python hadoop\mapreduce\main.py
```

**🎉 Chỉ cần lệnh này là đủ!** Hệ thống sẽ tự động thực hiện 4-stage MapReduce pipeline:
- **Stage 1**: Chuẩn bị và chia dữ liệu DTD (5,640 → 4×1,410 images)
- **Stage 2**: Map phase - CLIP-ViT-B/32 encoding song song
- **Stage 3**: Shuffle & Sort - Tổng hợp outputs từ mappers  
- **Stage 4**: Reduce phase - Chạy LAC/APS/RAPS algorithms
- **Visualization**: Tạo biểu đồ và lưu kết quả vào `MapReduceResult/`

---

## ⚙️ Luồng Xử Lý MapReduce Chi Tiết

### 🏗️ Kiến Trúc 4-Stage Pipeline

```
📊 INPUT: DTD Dataset (5,640 texture images, 47 classes)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ � STAGE 1: DATA PREPARATION & CHUNKING                    │
│ File: prepare_data.py                                       │
│ • Load DTD images from local_data/datasets/dtd/           │
│ • Create 4 balanced chunks (1,410 images each)            │
│ • Generate dtd_class_descriptions.json                     │
│ • Save chunks to hadoop_input/ directory                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ �️ STAGE 2: MAP PHASE - CLIP ENCODING                      │
│ File: clip_mapper.py                                        │
│ • Process 4 chunks in parallel (simulated)                │
│ • Load CLIP-ViT-B/32 model                                │
│ • Extract visual features for each image                   │
│ • Generate text embeddings for 47 texture classes         │
│ • Output: logits files + labels files                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ � STAGE 3: SHUFFLE & SORT                                 │
│ File: main.py (stage3_shuffle_sort)                        │
│ • Collect outputs from all 4 mappers                      │
│ • Aggregate logits: (4×1,410, 47) → (5,640, 47)          │
│ • Combine labels: (4×1,410,) → (5,640,)                   │
│ • Sort by image index for consistency                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ � STAGE 4: REDUCE PHASE - CONFORMAL PREDICTION            │
│ File: conformal_reducer.py                                 │
│ • Split: Calibration (2,820) + Test (2,820) sets         │
│ • Run 3 algorithms in parallel:                           │
│   ├── LAC (Least Ambiguous set-valued Classifier)        │
│   ├── APS (Adaptive Prediction Sets)                      │
│   └── RAPS (Regularized Adaptive Prediction Sets)        │
│ • Test multiple alpha values: 0.05, 0.1, 0.2             │
│ • Calculate coverage, set size, confidence metrics        │
└─────────────────────────────────────────────────────────────┘
    ↓
📊 OUTPUT: Results + Visualizations + Timing Data
```

### ⏱️ Timing & Performance Tracking

```
📊 Real-time Performance Monitoring:
├── Data Preparation: ~2.0s
├── Map Phase (CLIP): ~165s (dominant)
├── Shuffle & Sort: ~0.2s
└── Reduce Phase (CP): ~1.8s
Total Pipeline: ~170s (~2.8 minutes)
```

---

## 📊 Kết Quả Thu Được

### 📁 Thư Mục `MapReduceResult/`
Sau khi chạy xong, bạn sẽ có:

#### 1. 📈 Charts/[timestamp]/ - Biểu Đồ Trực Quan (7 biểu đồ)
- `01_coverage_rate_comparison.png` - So sánh tỷ lệ coverage
- `02_average_setsize_comparison.png` - So sánh kích thước set trung bình  
- `03_processing_time_comparison.png` - So sánh thời gian xử lý **THẬT**
- `04_coverage_setsize_tradeoff.png` - Đánh đổi coverage vs set size
- `05_temperature_scaling_analysis.png` - Phân tích temperature scaling
- `06_runtime_breakdown_analysis.png` - Phân tích chi tiết runtime
- `08_coverage_convergence.png` - Sự hội tụ của coverage

#### 2. 📄 Raw_Data/ - Dữ Liệu JSON
- `conformal_results_[timestamp].json` - Kết quả chi tiết với timing thật

**🎯 Tất cả dữ liệu biểu đồ đều THẬT 100%**: Coverage, set sizes, và runtime từ MapReduce pipeline thực tế!

---

## � Hiểu Các Biểu Đồ

### 🔍 Ý Nghĩa Từng Biểu Đồ

| Biểu đồ | Mục đích | Thông tin chính |
|---------|----------|-----------------|
| **01 - Coverage Rate** | So sánh độ che phủ | Xem thuật toán nào gần target 90% nhất |
| **02 - Set Size** | So sánh kích thước dự đoán | Thuật toán nào cho ít dự đoán nhất |
| **03 - Processing Time** | So sánh tốc độ | Thuật toán nào nhanh nhất |
| **04 - Tradeoff** | Cân bằng accuracy vs size | Vị trí tối ưu trên đồ thị |
| **05 - Temperature** | Ảnh hưởng nhiệt độ | Tham số nào cho kết quả tốt |
| **06 - Runtime Breakdown** | Chi tiết thời gian & hiệu suất | Phần nào tốn thời gian nhất |
| **07 - Conf-OT** | Tối ưu hóa post-processing | Transport cost vs coverage |
| **08 - Convergence** | Sự ổn định training-free | Coverage có ổn định không |

### 📊 Chỉ Số Quan Trọng

| Chỉ số | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| **Coverage Rate** | % dự đoán chứa đáp án đúng | Gần 90% |
| **Set Size** | Số lượng dự đoán trung bình | Càng nhỏ càng tốt |
| **Runtime** | Thời gian xử lý | Càng nhanh càng tốt |

### 📊 Kết Quả Mẫu (Dữ Liệu Thật từ DTD Dataset)
```
================================================================================
🎉 MAPREDUCE PIPELINE COMPLETED SUCCESSFULLY
⏱️  Total processing time: 174.43 seconds
⏱️  Data prep: 1.128s
⏱️  Map phase: 171.791s (CLIP encoding)
⏱️  Shuffle/Sort: 0.164s
⏱️  Reduce phase: 1.350s
================================================================================

📊 CONFORMAL PREDICTION RESULTS TABLE
================================================================================
Method       α = 0.10                       CCV↓ α = 0.05              CCV↓
               Top-1↑     Cov.    Size↓              Cov.    Size↓
--------------------------------------------------------------------------------
LAC              42.0    0.913     12.2     0.09    0.957     18.4     0.05
APS              42.0    0.954     19.0     0.08    0.984     25.1     0.04
RAPS             42.0    0.904     12.9     0.09    0.951     18.8     0.05
================================================================================
Notes:
- Top-1↑: Higher is better (accuracy)
- Cov.: Coverage rate (should be ≥ 1-α)
- Size↓: Average prediction set size (lower is better)
- CCV↓: Conditional Coverage Violation (lower is better)
================================================================================

📊 EXPORTING RESULTS TO EXCEL
----------------------------------------
✅ Excel file created: C:\BigData\CLIP-Conformal\MapReduceResult\Reports\11h32am_20251020.xlsx
📁 Location: C:\BigData\CLIP-Conformal\MapReduceResult\Reports
📊 Sheets: Research_Comparison, Summary, Detailed_Results, Metadata

[+] Results saved in: MapReduceResult/Raw_Data/conformal_results_20251020_113224.json
[+] Charts created in: MapReduceResult/Charts/11h32am_20-10-2025_charts/
```

**Giải thích Kết Quả**:
- **LAC**: Đơn giản, nhanh, coverage tốt cho cả 2 alpha (91.3%→95.7%)
- **APS**: Coverage cao nhất (95.4%→98.4%) nhưng set size lớn nhất
- **RAPS**: Cân bằng tốt, ổn định với CCV thấp

### 🆕 Output Files Chi Tiết

#### 📁 MapReduceResult/Reports/[timestamp].xlsx
**4 Excel Sheets được tạo**:
1. **Research_Comparison**: Bảng so sánh theo format nghiên cứu khoa học
2. **Summary**: Tóm tắt chi tiết với status và phân tích  
3. **Detailed_Results**: Dữ liệu raw cho phân tích thống kê
4. **Metadata**: Thông tin về dataset, model, pipeline settings

#### 📁 MapReduceResult/Charts/[timestamp]_charts/
**7 Biểu Đồ Visualization**:
1. `01_coverage_rate_comparison.png` - So sánh coverage rate
2. `02_average_setsize_comparison.png` - So sánh kích thước prediction sets
3. `03_processing_time_comparison.png` - So sánh thời gian xử lý
4. `04_coverage_setsize_tradeoff.png` - Trade-off coverage vs set size
5. `05_temperature_scaling_analysis.png` - Phân tích temperature scaling
6. `06_runtime_breakdown_analysis.png` - Phân tích chi tiết runtime
7. `08_coverage_convergence.png` - Phân tích hội tụ coverage

#### 📄 MapReduceResult/Raw_Data/conformal_results_[timestamp].json
**Structure JSON**:
```json
{
  "0.1": {
    "LAC": {"coverage": 0.913, "avg_set_size": 12.17, "ccv": 0.09, ...},
    "APS": {"coverage": 0.954, "avg_set_size": 18.99, "ccv": 0.08, ...},
    "RAPS": {"coverage": 0.904, "avg_set_size": 12.88, "ccv": 0.09, ...}
  },
  "0.05": {
    "LAC": {"coverage": 0.957, "avg_set_size": 18.38, "ccv": 0.05, ...},
    "APS": {"coverage": 0.984, "avg_set_size": 25.11, "ccv": 0.04, ...},
    "RAPS": {"coverage": 0.951, "avg_set_size": 18.76, "ccv": 0.05, ...}
  }
}
```

---

## 🔧 Xử Lý Lỗi Thường Gặp

### ❌ Lỗi: "Module not found"
```bash
# Cài đặt thư viện thiếu
pip install torch numpy pandas matplotlib openpyxl
```

### 🆕 ❌ Lỗi: "pandas not available for Excel export"
```bash
# Cài đặt pandas và openpyxl cho Excel export
pip install pandas openpyxl
```

### ❌ Lỗi: "File not found"  
```bash
# Đảm bảo bạn ở đúng thư mục
cd C:\BigData\CLIP-Conformal
pwd  # Kiểm tra đường dẫn hiện tại
```

### 🆕 ❌ Lỗi: "Excel file already open"
```bash
# Đóng file Excel đang mở trong Microsoft Excel
# Hoặc đổi tên file trong function export_results_to_excel()
```

### ❌ Lỗi: "Permission denied"
```bash
# Chạy với quyền admin hoặc thay đổi quyền thư mục
```

### 🆕 ❌ Charts không được tạo
```bash
# Kiểm tra matplotlib backend
pip install matplotlib
# Nếu vẫn lỗi, kiểm tra đường dẫn output_dir trong draw_charts.py
```

---

## 🎯 Tùy Chỉnh Tham Số

### 🆕 Thay Đổi Alpha Values (Dual-Alpha Testing)
Trong file `conformal_reducer.py`, tìm dòng:
```python
alpha_values = [0.1, 0.05]  # α = 0.10 (90% coverage) và α = 0.05 (95% coverage)
```
**Tùy chỉnh**:
- `[0.1]` → Chỉ test α = 0.10 (90% confidence)
- `[0.05]` → Chỉ test α = 0.05 (95% confidence)  
- `[0.1, 0.05, 0.01]` → Test 3 alpha values (90%, 95%, 99% confidence)

### Thay Đổi Excel Export Format
Trong file `main.py`, function `export_results_to_excel()`:
```python
# Tùy chỉnh filename format
excel_filename = f"{hour}h{minute}{ampm}_{date}.xlsx"
# Có thể thay đổi thành:
excel_filename = f"conformal_results_{date}_{hour}{minute}.xlsx"
```

### Thay Đổi Dataset
```python
# Trong hàm load_dtd_data(), thay đổi đường dẫn
cache_file = 'local_data/your_dataset.npz'
```

### Chạy Module Vẽ Biểu Đồ Riêng
```bash
# Chạy file draw_charts.py độc lập để test
cd hadoop\mapreduce
python draw_charts.py
```
**Chức năng**: Tạo biểu đồ test với dữ liệu mẫu để kiểm tra hệ thống vẽ biểu đồ.

### Thay Đổi Cấu Hình MapReduce
```python
# Trong prepare_data.py - thay đổi chunk size
chunk_size = 1410  # Số images per chunk (default)

# Trong main.py - thay đổi split ratio
split_ratio = 0.5  # 50% calibration, 50% test
```

---

## 🆕 Giải Thích Metrics Mới

### 📊 Enhanced Metrics Table
| Metric | Ký Hiệu | Giải Thích | Mục Tiêu |
|--------|---------|------------|----------|
| **Top-1 Accuracy** | Top-1↑ | Độ chính xác dự đoán đơn | Càng cao càng tốt |
| **Coverage Rate** | Cov. | Tỷ lệ true label nằm trong prediction set | ≥ (1-α) |
| **Average Set Size** | Size↓ | Kích thước trung bình prediction set | Càng nhỏ càng tốt |
| **CCV** | CCV↓ | Conditional Coverage Violation | Càng thấp càng tốt |

### 🎯 Dual-Alpha Comparison
- **α = 0.10**: Target coverage 90% - Nhanh, set size nhỏ, ít conservative
- **α = 0.05**: Target coverage 95% - Chậm hơn, set size lớn hơn, conservative hơn

### 📈 Performance Interpretation
**Ideal Method**: High Top-1↑, Coverage ≥ target, Low Size↓, Low CCV↓
- **LAC**: Tốt cho applications cần tốc độ
- **APS**: Tốt cho applications cần coverage cao  
- **RAPS**: Cân bằng tốt giữa coverage và efficiency

---

## 📚 Thuật Ngữ Đơn Giản

| Thuật ngữ | Giải thích đơn giản |
|-----------|-------------------|
| **Conformal Prediction** | Phương pháp dự đoán "an toàn" với độ tin cậy |
| **Coverage** | Tỷ lệ dự đoán có chứa đáp án đúng |
| **Set Size** | Số lượng đáp án có thể trong mỗi dự đoán |
| **Calibration** | Quá trình "học" để hiểu độ tin cậy |
| **DTD Dataset** | Bộ dữ liệu 47 loại texture khác nhau |
| **🆕 Top-1 Accuracy** | Độ chính xác dự đoán lựa chọn đầu tiên |
| **🆕 CCV** | Mức độ vi phạm coverage theo từng class |
| **🆕 Dual-Alpha** | Kiểm thử với 2 mức độ tin cậy (90% & 95%) |
| **🆕 Excel Export** | Xuất kết quả ra file Excel đa sheet |

---

**🎉 Chúc bạn chạy dự án thành công!** 

Mỗi lần chạy sẽ tạo folder charts mới với timestamp để dễ theo dõi. Tất cả kết quả đều được lưu chi tiết trong `MapReduceResult/`.