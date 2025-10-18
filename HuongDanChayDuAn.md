# 🎯 Hướng Dẫn Chạy Dự Án CLIP-Conformal Prediction

## 📋 Mục Đích Dự Án
Dự án này sử dụng thuật toán **Conformal Prediction** để dự đoán texture (kết cấu bề mặt) của hình ảnh với độ tin cậy cao. Thay vì chỉ đưa ra 1 kết quả dự đoán, hệ thống sẽ đưa ra một **tập kết quả có thể** với mức độ tin cậy 90%.

**Ví dụ đơn giản**: 
- Dự đoán thông thường: "Đây là texture gỗ" 
- Conformal Prediction: "Đây có thể là texture gỗ hoặc vải, với độ tin cậy 90%"

---

## 📁 Cấu Trúc Dự Án

### 🗂️ Thư Mục Chính

```
CLIP-Conformal/
├── 📁 hadoop/MapReduce/          # Thư mục chính chứa code
│   ├── 📄 main.py               # FILE CHÍNH ĐỂ CHẠY
│   └── 📄 draw_charts.py        # Module vẽ biểu đồ (tách riêng)
├── 📁 local_data/               # Dữ liệu đã chuẩn bị sẵn
│   ├── 📁 datasets/dtd/         # Hình ảnh texture DTD
│   └── 📄 dtd_clip-vit-b_32.npz # Dữ liệu đã xử lý sẵn
├── 📁 conformal/               # Thuật toán Conformal Prediction
├── 📁 data/                    # Các hàm xử lý dữ liệu
├── 📁 MapReduceResult/         # KẾT QUẢ CUỐI CÙNG (tự động tạo)
│   ├── 📁 Charts/              # Biểu đồ PNG
│   ├── 📁 Raw_Data/            # Dữ liệu JSON
│   └── 📁 Reports/             # Báo cáo Excel
└── 📄 HuongDanChayDuAn.md     # File hướng dẫn này
```

### 📄 Mục Đích Từng File Quan Trọng

| File/Thư mục | Chức năng |
|---------------|-----------|
| `hadoop/MapReduce/main.py` | **FILE CHÍNH** - Chạy toàn bộ hệ thống |
| `hadoop/MapReduce/draw_charts.py` | **Module vẽ biểu đồ** - Tách riêng để dễ quản lý |
| `local_data/dtd_clip-vit-b_32.npz` | Dữ liệu 1,692 hình ảnh đã xử lý sẵn |
| `conformal/conformal_methods.py` | Các thuật toán LAC, APS, RAPS |
| `MapReduceResult/` | **Thư mục kết quả** chứa tất cả output |

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
python hadoop\MapReduce\main.py
```

**🎉 Chỉ cần lệnh này là đủ!** Hệ thống sẽ tự động:
- Đọc dữ liệu DTD (1,692 hình ảnh texture)
- Chạy 3 thuật toán: LAC, APS, RAPS
- Tạo biểu đồ và báo cáo
- Lưu kết quả vào thư mục `MapReduceResult/`

---

## ⚙️ Luồng Xử Lý MapReduce

### 🗺️ Giai Đoạn MAP (Phân tán xử lý)
```
📊 Dữ liệu DTD (1,692 hình ảnh)
    ↓
🔄 Chia thành 2 phần:
    ├── 📈 Calibration Set (846 mẫu) - Để "học" độ tin cậy
    └── 📊 Test Set (846 mẫu) - Để kiểm tra kết quả
```

### 🔄 Giai Đoạn REDUCE (Tổng hợp kết quả)
```
🧮 Chạy song song 3 thuật toán:
    ├── 🔵 LAC (Least Ambiguous set-valued Classifier)
    ├── 🟣 APS (Adaptive Prediction Sets) 
    └── 🟡 RAPS (Regularized Adaptive Prediction Sets)
    ↓
📊 Tính toán metrics:
    ├── Coverage Rate (Tỷ lệ dự đoán đúng)
    ├── Set Size (Kích thước tập dự đoán)
    └── Runtime (Thời gian xử lý)
```

---

## 📊 Kết Quả Thu Được

### 📁 Thư Mục `MapReduceResult/`
Sau khi chạy xong, bạn sẽ có:

#### 1. 📈 Charts/ - Biểu Đồ Trực Quan
- `01_coverage_rate_comparison.png` - So sánh tỷ lệ coverage
- `02_average_setsize_comparison.png` - So sánh kích thước set trung bình
- `03_processing_time_comparison.png` - So sánh thời gian xử lý
- `04_coverage_setsize_tradeoff.png` - Đánh đổi coverage vs set size
- `05_temperature_scaling_analysis.png` - Phân tích temperature scaling
- `06_runtime_breakdown_analysis.png` - Phân tích chi tiết runtime
- `07_confot_optimization.png` - Tối ưu hóa Conf-OT
- `08_coverage_convergence.png` - Sự hội tụ của coverage

#### 2. 📄 Raw_Data/ - Dữ Liệu Thô
- `results.json` - Kết quả đơn giản, dễ đọc
- `conformal_prediction_results.json` - Kết quả chi tiết

#### 3. 📊 Reports/ - Báo Cáo Excel
- `DTD_Conformal_Prediction_Results_[timestamp].xlsx` - Báo cáo đầy đủ

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

### 📊 Kết Quả Mẫu
```
✅ LAC : Coverage=43.1%, Size=1.0, Time=0.01s
✅ APS : Coverage=42.8%, Size=1.1, Time=0.01s  
✅ RAPS: Coverage=56.5%, Size=2.0, Time=0.01s
```

**Giải thích**:
- RAPS có độ che phủ cao nhất (56.5%) nhưng đưa ra nhiều dự đoán (2.0)
- LAC và APS nhanh hơn và đưa ra ít dự đoán hơn (~1.0)

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
cd hadoop\MapReduce
python draw_charts.py
```
**Chức năng**: Tạo biểu đồ test với dữ liệu mẫu để kiểm tra hệ thống vẽ biểu đồ.

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

## ✅ Checklist Hoàn Thành

- [ ] Đã cài đặt Python và các thư viện
- [ ] Đã tải về đầy đủ dữ liệu DTD  
- [ ] Chạy thành công `python hadoop\MapReduce\main.py`
- [ ] Thấy thư mục `MapReduceResult/` được tạo
- [ ] Có 4 file PNG trong `Charts/`
- [ ] Có file Excel trong `Reports/`
- [ ] Có 2 file JSON trong `Raw_Data/`

---

## 🆘 Hỗ Trợ

### Nếu gặp vấn đề:
1. **Kiểm tra**: File `dtd_clip-vit-b_32.npz` có tồn tại không?
2. **Thử lại**: Xóa thư mục `MapReduceResult/` và chạy lại
3. **Xem log**: Đọc thông báo lỗi trên terminal để hiểu vấn đề

### Thông tin hệ thống:
- **Dataset**: DTD (Describable Textures Dataset) - 1,692 ảnh
- **Classes**: 47 loại texture (gỗ, vải, kim loại, etc.)
- **Target Coverage**: 90% (có thể điều chỉnh)
- **Methods**: LAC, APS, RAPS

---

**🎉 Chúc bạn chạy dự án thành công!** 

Nếu có thắc mắc, hãy xem kỹ các file kết quả trong `MapReduceResult/` - tất cả thông tin đều được lưu chi tiết ở đó.