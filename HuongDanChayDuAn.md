# 🎯 Hướng Dẫn Chạy Dự Án CLIP-Conformal Prediction

## 📋 Mục Đích Dự Án
Dự án này sử dụng thuật toán **Conformal Prediction** để dự đoán texture (kết cấu bề mặt) của hình ảnh với độ tin cậy cao. Thay vì chỉ đưa ra 1 kết quả dự đoán, hệ thống sẽ đưa ra một **tập kết quả có thể** với mức độ tin cậy 90%.

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
│   │   └── 📁 [HHhmm_DD-MM-YYYY_charts]/ # Session-specific charts
│   └── 📁 Raw_Data/            # JSON results storage
│       └── � conformal_results_*.json   # Detailed results
└── 📄 HuongDanChayDuAn.md     # This guide
```

### 📄 Luồng Xử Lý MapReduce - File Mapping

| Stage | File | Input | Output | Chức năng |
|-------|------|-------|--------|-----------|
| **Stage 1** | `prepare_data.py` | DTD images (5,640) | Chunks (4×1,410) | Data chunking & preparation |
| **Stage 2** | `clip_mapper.py` | Data chunks | CLIP logits | Parallel CLIP-ViT-B/32 encoding |
| **Stage 3** | `main.py` | Mapper outputs | Sorted data | Shuffle & sort aggregation |
| **Stage 4** | `conformal_reducer.py` | Aggregated data | CP results | LAC/APS/RAPS algorithms |
| **Visualization** | `draw_charts.py` | CP results | Charts | Real-time visualization |

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
⏱️  Total processing time: 169.04 seconds
⏱️  Data prep: 2.474s
⏱️  Map phase: 303.204s (CLIP encoding)
⏱️  Shuffle/Sort: 0.225s
⏱️  Reduce phase: 4.498s
================================================================================

📊 CONFORMAL PREDICTION RESULTS SUMMARY:
✅ LAC : Coverage=91.3%, Size=12.2, Runtime=0.70s
✅ APS : Coverage=95.4%, Size=19.0, Runtime=1.40s  
✅ RAPS: Coverage=90.4%, Size=12.9, Runtime=1.75s

[+] Results saved in: MapReduceResult/Raw_Data/
[+] Charts created in: MapReduceResult/Charts/11h32am_18-10-2025_charts/
```

**Giải thích**:
- **LAC**: Đơn giản, nhanh (0.70s), coverage tốt (91.3%), set size nhỏ (12.2)
- **APS**: Coverage cao nhất (95.4%) nhưng set size lớn (19.0)  
- **RAPS**: Cân bằng tốt, complexity cao nhất (1.75s)

---

## 🔧 Xử Lý Lỗi Thường Gặp

### ❌ Lỗi: "Module not found"
```bash
# Cài đặt thư viện thiếu
pip install torch numpy pandas matplotlib openpyxl
```

### ❌ Lỗi: "File not found"  
```bash
# Đảm bảo bạn ở đúng thư mục
cd C:\BigData\CLIP-Conformal
pwd  # Kiểm tra đường dẫn hiện tại
```

### ❌ Lỗi: "Permission denied"
```bash
# Chạy với quyền admin hoặc thay đổi quyền thư mục
```

---

## 🎯 Tùy Chỉnh Tham Số

### Thay Đổi Độ Tin Cậy
Trong file `main.py`, tìm dòng:
```python
alpha = 0.1  # Độ tin cậy 90% (1 - 0.1)
```
- `alpha = 0.05` → Độ tin cậy 95%
- `alpha = 0.2` → Độ tin cậy 80%

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

## 📚 Thuật Ngữ Đơn Giản

| Thuật ngữ | Giải thích đơn giản |
|-----------|-------------------|
| **Conformal Prediction** | Phương pháp dự đoán "an toàn" với độ tin cậy |
| **Coverage** | Tỷ lệ dự đoán có chứa đáp án đúng |
| **Set Size** | Số lượng đáp án có thể trong mỗi dự đoán |
| **Calibration** | Quá trình "học" để hiểu độ tin cậy |
| **DTD Dataset** | Bộ dữ liệu 47 loại texture khác nhau |

---

**🎉 Chúc bạn chạy dự án thành công!** 

Mỗi lần chạy sẽ tạo folder charts mới với timestamp để dễ theo dõi. Tất cả kết quả đều được lưu chi tiết trong `MapReduceResult/`.