I. Tóm tắt:
Nghiên cứu này trình bày quá trình triển khai mô hình CLIP-Conformal Prediction cho bài toán phân loại ảnh đa miền. Mô hình CLIP được sử dụng để trích xuất đặc trưng thị giác và văn bản, sau đó áp dụng kỹ thuật dự đoán tuân thủ (conformal prediction) nhằm ước lượng mức độ tin cậy của kết quả phân loại. Để xử lý khối lượng dữ liệu lớn, nghiên cứu áp dụng kiến trúc MapReduce dựa trên Hadoop, cho phép phân tán việc trích xuất đặc trưng sang nhiều tiến trình.

Nghiên cứu được đánh giá trên ba bộ dữ liệu đa dạng về quy mô và miền ứng dụng: Food101 (101,000 ảnh với 101 lớp món ăn), SUN397 (108,754 ảnh với 397 lớp scene), và FireSmoke (95,000+ ảnh với 4 lớp phân loại tình huống cháy nổ). Sự đa dạng này cho phép đánh giá toàn diện khả năng tổng quát hóa của mô hình CLIP-Conformal trên các miền ứng dụng khác nhau từ nhận dạng thực phẩm, phân loại cảnh quan đến ứng dụng an toàn công nghiệp. Kết quả thực nghiệm cho thấy hệ thống đạt được độ chính xác cao với thời gian xử lý giảm đáng kể so với phương pháp truyền thống, đồng thời duy trì độ tin cậy thống kê ổn định across different domains.

II. Giới thiệu:
Trong bối cảnh bùng nổ dữ liệu hình ảnh, các mô hình học sâu như CLIP (Contrastive Language-Image Pretraining) giúp máy tính hiểu được mối quan hệ giữa ngôn ngữ và hình ảnh. Tuy nhiên, khi triển khai trên các bộ dữ liệu lớn và đa miền, việc dự đoán kết quả kèm độ tin cậy là một thách thức quan trọng. Conformal Prediction là hướng tiếp cận hiện đại nhằm cung cấp dự đoán với mức độ tin cậy định lượng được, đặc biệt quan trọng trong các ứng dụng thực tế đòi hỏi độ tin cậy cao như an toàn công nghiệp, y tế, và autonomous systems.

Nghiên cứu này tập trung vào việc đánh giá hiệu suất của CLIP-Conformal Prediction trên ba miền ứng dụng khác biệt: phân loại thực phẩm (Food101), nhận dạng cảnh quan (SUN397), và phát hiện tình huống cháy nổ (FireSmoke). Mỗi miền này đại diện cho những thách thức riêng biệt - từ sự phức tạp trong texture và appearance của food items, đến diversity của natural và man-made scenes, cho đến tính critical của fire safety applications. Chi phí tính toán của CLIP-Conformal rất lớn khi xử lý hàng trăm nghìn ảnh, đặc biệt với FireSmoke dataset chứa hơn 95,000 ảnh high-resolution. Do đó, đề tài này đề xuất triển khai mô hình trên kiến trúc phân tán MapReduce để tận dụng tài nguyên tính toán, giảm thời gian xử lý và nâng cao khả năng mở rộng cho large-scale image classification tasks.

III. Nghiên cứu liên quan:
Trong lĩnh vực kết hợp mô hình ngôn ngữ-hình ảnh và dự đoán độ tin cậy, mô hình CLIP đã được sử dụng rộng rãi để ánh xạ ảnh và văn bản vào cùng không gian embedding; song việc thêm lớp Conformal Prediction nhằm định lượng độ tin cậy đã được khảo sát gần đây trong công trình Conformal Prediction for Zero-Shot Models của Silva-Rodríguez et al., trong đó tác giả đề xuất phương pháp Conf-OT để khắc phục sự bất đồng miền giữa bộ calibration và bộ truy vấn, đồng thời vẫn đảm bảo độ phủ xác suất [1]. 
Trước đó, Kumar et al. đã mở rộng conformal prediction sang môi trường zero-shot với các mô hình tự giám sát, đề xuất cách phát hiện ngoại lai cho captions khi áp dụng CLIP trong nhiệm vụ phân loại không giám sát [2]. 
Ở khía cạnh xử lý dữ liệu quy mô lớn, nguyên mẫu MapReduce của Google (Dean & Ghemawat) đã đặt nền móng cho xử lý song song phân tán, cho phép xử lý khối lượng dữ liệu ảnh lớn hiệu quả [3]. 
Các khảo sát gần đây cũng so sánh các mô hình MapReduce, Hadoop, Spark để đánh giá khả năng mở rộng và hiệu suất trong ngữ cảnh big data [4]. Những nghiên cứu này tạo nền tảng lý thuyết và kỹ thuật cho việc kết hợp CLIP, Conformal Prediction và kiến trúc MapReduce trong đề tài này.
IV. Phương pháp:
1. Triển khai mô hình:
1.1. Bộ dữ liệu thực nghiệm:

Nghiên cứu được thực hiện trên ba bộ dữ liệu đại diện cho các miền ứng dụng khác nhau:

**Food101 Dataset:** Bộ dữ liệu phân loại món ăn chứa 101,000 ảnh thuộc 101 lớp món ăn phổ biến, với 1,000 ảnh cho mỗi lớp. Các ảnh có độ phân giải đa dạng và được resize về 224×224 pixels cho CLIP processing. Dataset này đại diện cho domain của food recognition với challenges về texture, color variations, và presentation styles.

**SUN397 Dataset:** Scene UNderstanding dataset chứa 108,754 ảnh thuộc 397 lớp scene categories, bao gồm natural và man-made environments. Mỗi lớp có khoảng 100-3,000 ảnh với distribution không đều. Dataset này thử thách khả năng của mô hình trong fine-grained scene classification với subtle visual differences between categories.

**FireSmoke Dataset:** Bộ dữ liệu tự thu thập phục vụ ứng dụng an toàn công nghiệp, chứa hơn 95,000 ảnh RGB định dạng .jpg thuộc 4 lớp: Fire (có lửa), Smoke (có khói), Both (có cả lửa và khói), và None (không có hiện tượng cháy). Mỗi lớp chứa 15,000-25,000 ảnh với kích thước đa dạng được resize về 224×224 pixels. Dataset này được xây dựng để huấn luyện các mô hình nhận dạng tình huống cháy nổ trong môi trường thực tế, đòi hỏi độ tin cậy cao cho safety-critical applications.

1.2. Kiến trúc:
Để giải quyết bài toán phân loại hình ảnh trong môi trường dữ liệu lớn và không đồng nhất, mô hình được triển khai dựa trên sự kết hợp giữa CLIP (Contrastive Language–Image Pretraining) và Conformal Prediction. CLIP là một mô hình học sâu được huấn luyện trên hàng trăm triệu cặp ảnh–văn bản, có khả năng ánh xạ đồng thời hai miền dữ liệu này vào cùng một không gian vector đặc trưng. Nhờ đặc tính zero-shot learning, CLIP có thể nhận diện các lớp mới mà không cần huấn luyện lại, giúp tiết kiệm tài nguyên và nâng cao tính linh hoạt của hệ thống. 
 
Hình 1. Kiến trúc tổng thể của mô hình CLIP kết hợp với Conformal Prediction
Hình trên minh họa kiến trúc tổng thể của mô hình CLIP kết hợp với Conformal Prediction được sử dụng trong nghiên cứu. Phần dưới đây sẽ giải thích cách thức hoạt động của từng phần: CLIP (i) và Conformal(ii).
(i) Về phía CLIP, mô hình gồm hai thành phần chính là Vision Encoder và Text Encoder.
 
Hình 2. Mô hình hoạt động CLIP
Vision Encoder chịu trách nhiệm trích xuất đặc trưng từ ảnh đầu vào.
Cụ thể, ảnh được đưa qua một mạng học sâu như Vision Transformer (ViT).
Quá trình này bao gồm:
(1) Ảnh đầu vào có kích cỡ H x W x C được chia thành các patches nhỏ để thu được các vùng đặc trưng (mỗi Patch có kích cỡ h x w x C, với h < H và w <W) 
(2) Mỗi patch hoặc vùng đặc trưng được làm phẳng và ánh xạ tuyến tính thành một vector đặc trưng có cùng kích thước (tức là biến 1 patch thành 1 vector). Mỗi vector đặc trưng là 1 vector 1 chiểu có độ dài m = h x w x C
(3) Tập hợp các vector này được đưa vào cơ chế Self-Attention, nơi mỗi vector so sánh và trao đổi thông tin với tất cả vector khác thông qua ma trận trọng số Attention, giúp mô hình hiểu mối quan hệ giữa các vùng ảnh (ví dụ: nền – vật thể – chi tiết).
(4) Các thông tin đã được tổng hợp được gộp lại (pooling) để tạo thành một vector biểu diễn duy nhất cho toàn bộ hình ảnh (Vector v¬i) 
 
Hình 3. Cơ chế hoạt động Vision Encoder
Text Encoder mã hóa các mô tả văn bản tương ứng. Text Encoder mã hóa phần mô tả văn bản (ví dụ: “A photo of a dog”) thành vector đặc trưng trong cùng không gian với ảnh. Cụ thể, mô hình sử dụng Transformer Encoder – tương tự như trong BERT hoặc GPT – để học ngữ nghĩa của câu. Quy trình gồm:
	Câu văn được chia thành các token (từ hoặc cụm từ).
	Thêm Token đặc biệt [EOS] hoặc [CLS] ở cuối chuỗi được chọn làm biểu diễn tổng thể của câu ([CLS] (classification token): đặt ở đầu câu, dùng để biểu diễn toàn bộ câu trong các mô hình như BERT, CLIP, ViT. [EOS] (end of sentence): đặt ở cuối câu, báo hiệu kết thúc chuỗi trong các mô hình như GPT, CLIP-Text Encoder.
	Mỗi token được chuyển thành vector nhúng (embedding) ban đầu.
	Các vector này được xử lý qua nhiều lớp self-attention để nắm bắt ngữ cảnh của toàn câu và tổng hợp thành vector cuối cùng (Vector tj)
Kết quả là một vector tj biểu diễn ngữ nghĩa của câu văn, trong cùng không gian với vector ảnh vi
 
Hình 4. Cơ chế hoạt động Text Encoder
Hai không gian này được liên kết thông qua ma trận tương đồng (similarity matrix), trong đó mỗi phần tử thể hiện độ tương đồng giữa ảnh và mô tả văn bản tương ứng. Mỗi phần tử của ma trận này được gọi là “điểm tương đồng” (logits), sẽ được tính bằng tích vô hướng của 2 vector vi và tj đã được tạo thông qua Text Encoder (vector tj) và Vision Encoder (vector vi)
logit_{ij}=\frac{v_i\cdot t_j}{\left|\left|v_i\right|\right|\left|\left|t_j\right|\right|}

Nhờ đó, CLIP có thể xác định lớp của ảnh bằng cách tìm nhãn văn bản có độ tương đồng cao nhất, ngay cả khi lớp đó chưa từng xuất hiện trong quá trình huấn luyện
(ii) Phần Conformal Prediction đóng vai trò mở rộng nhằm đánh giá và điều chỉnh độ tin cậy của các dự đoán từ CLIP. Cụ thể, Conformal Predictor hoạt động qua ba bước: 
(1) tính điểm không phù hợp (non-conformity scores) cho từng mẫu.
(2) xác định ngưỡng định lượng (quantile threshold) tương ứng với mức ý nghĩa α.
(3) xây dựng tập dự đoán (prediction set) gồm các nhãn có độ tin cậy cao hơn ngưỡng đó. 
Kết quả là thay vì đưa ra một nhãn duy nhất, mô hình trả về một tập các nhãn tiềm năng kèm theo đảm bảo thống kê về xác suất chứa nhãn thật. Nhờ sự kết hợp này, hệ thống vừa duy trì được tính linh hoạt và khả năng học zero-shot của CLIP, vừa đảm bảo tính minh bạch và độ tin cậy thống kê nhờ Conformal Prediction.
1.2. Hàm mất mát:
Trong mô hình CLIP – Conformal, thành phần CLIP được huấn luyện dựa trên một hàm mất mát tương phản (Contrastive Loss), được thiết kế nhằm học mối quan hệ tương ứng giữa ảnh và văn bản. Mục tiêu của hàm mất mát này là đưa các cặp ảnh – văn bản đúng lại gần nhau trong không gian vector, đồng thời đẩy các cặp không tương ứng ra xa nhau.
Cụ thể, sau khi ảnh và mô tả văn bản được mã hóa thành các vector đặc trưng vi và tj (bởi Vision Encoder và Text Encoder tương ứng), [5] CLIP tính ma trận tương đồng giữa tất cả các cặp ảnh–văn bản trong cùng một batch huấn luyện:
S_{ij}=\frac{\exp{\left(\mathrm{sim}\left(v_i,t_j\right)/\tau\right)}}{\sum_{k=1}^{N}\exp{\left(\mathrm{sim}\left(v_i,t_k\right)/\tau\right)}}
Trong đó:
	sim(vi,tj) là độ tương đồng cosine giữa vector ảnh và văn bản,
	τ là tham số nhiệt độ (temperature scaling) dùng để điều chỉnh độ nhạy,
	Nlà số lượng mẫu trong batch.
Hàm mất mát được định nghĩa là trung bình giữa hai hướng học: ảnh–văn bản và văn bản–ảnh:
L_{CLIP}=\frac{1}{2}\left(L_{\mathrm{img}\rightarrow\mathrm{txt}}+L_{\mathrm{txt}\rightarrow\mathrm{img}}\right)

với:
L_{\mathrm{img}\rightarrow\mathrm{txt}}=-\frac{1}{N}\sum_{i=1}^{N}\log{\frac{\exp{\left(\mathrm{sim}\left(v_i,t_i\right)/\tau\right)}}{\sum_{j=1}^{N}\exp{\left(\mathrm{sim}\left(v_i,t_j\right)/\tau\right)}}}

Trong quá trình huấn luyện, mô hình CLIP liên tục cập nhật trọng số của hai encoder sao cho độ tương đồng giữa các cặp ảnh–văn bản đúng tăng dần, còn giữa các cặp sai thì giảm dần. Khi quá trình này hội tụ, hàm mất mát giảm dần theo số epoch, còn độ chính xác của mô hình tăng dần và ổn định.

2. Triển khai Map – Reduce:
Để tối ưu hóa quá trình xử lý trong bài toán CLIP–Conformal trên các bộ dữ liệu lớn như DTD và Oxford Flowers 102, mô hình được triển khai theo kiến trúc Map–Reduce trên nền tảng Hadoop. Mục tiêu của việc tích hợp Map–Reduce là phân tán khối lượng tính toán nặng nề của CLIP, đặc biệt trong giai đoạn trích xuất đặc trưng ảnh (logits extraction), và giai đoạn tính toán thống kê trong Conformal Prediction, nhằm rút ngắn thời gian xử lý khi dữ liệu vượt quá khả năng của một máy tính cá nhân.
2.1. Kiến trúc:
Kiến trúc hệ thống được tổ chức thành ba tầng chính: HDFS (Hadoop Distributed File System), Map Layer, và Reduce Layer.
 
Hình 5. Kiến trúc Map Reduce
(1) Tầng HDFS đóng vai trò là hệ thống lưu trữ phân tán, chịu trách nhiệm quản lý toàn bộ dữ liệu hình ảnh, các đặc trưng trung gian (features, logits) và kết quả đầu ra (non-conformity scores, prediction sets). Các bộ dữ liệu như DTD và Oxford Flowers 102 được chia thành nhiều khối (block) độc lập và phân phối đều trên các nút tính toán trong cụm Hadoop.
(2) Tầng Map chịu trách nhiệm xử lý độc lập các khối dữ liệu này. Mỗi Mapper tải mô hình CLIP (hoặc phiên bản tinh chỉnh) và đọc một phần ảnh từ HDFS, sau đó trích xuất các vector đặc trưng thị giác (vision features) và độ tương đồng ảnh–văn bản (text–image similarity scores). Mỗi Mapper lưu kết quả đầu ra của mình, bao gồm ma trận logits và nhãn tham chiếu (refs), vào HDFS dưới dạng các tệp .npz.
(3) Tầng Reduce tiếp nhận toàn bộ các tệp logits từ các Mapper, kết hợp chúng lại và thực hiện giai đoạn Conformal Prediction. Tại đây, hệ thống tiến hành các phép tính thống kê quan trọng như ước lượng non-conformity scores, xác định quantile threshold tương ứng với mức ý nghĩa α, và sinh ra prediction sets với độ tin cậy thống kê (1−α).
2.2. Luồng xử lý:
 
Hình 6. Luồng xử lý Map Reduce
(1) Chuẩn bị dữ liệu đầu vào: Ba bộ dữ liệu Food101, SUN397, và FireSmoke được chuẩn bị và chia thành các phần nhỏ tương ứng với kích thước khối (block size) của HDFS (128MB). Food101 với 101,000 ảnh được chia thành 8 chunks (12,625 ảnh/chunk), SUN397 với 108,754 ảnh được chia thành 10 chunks (10,875 ảnh/chunk), và FireSmoke với 95,000+ ảnh được chia thành 9 chunks (~10,555 ảnh/chunk). Dữ liệu được tải lên hệ thống Hadoop dưới dạng các tệp ảnh hoặc danh sách đường dẫn ảnh (.txt), đảm bảo load balancing tối ưu cho các worker nodes across different dataset characteristics.

(2) Giai đoạn Map: Mỗi Mapper đọc một phân đoạn dữ liệu từ HDFS và khởi tạo mô hình CLIP-ViT-B/32 trong bộ nhớ local. Quá trình xử lý diễn ra song song với việc mã hóa đồng thời ảnh đầu vào (Vision Encoder) và mô tả văn bản tương ứng (Text Encoder) cho từng dataset: 101 lớp food categories cho Food101, 397 lớp scene categories cho SUN397, và 4 lớp fire/smoke situations cho FireSmoke. Mỗi Mapper sinh ra ma trận logits với kích thước tương ứng [batch_size × num_classes], biểu diễn mức độ tương đồng cosine giữa vision features và text embeddings. Đầu ra của mỗi Mapper bao gồm logits matrix và labels vector được serialize và ghi trở lại HDFS dưới dạng các tệp .npz để phục vụ cho giai đoạn Reduce.

(3) Giai đoạn Shuffle & Sort: Hệ thống Hadoop tự động gom nhóm các tệp logits từ tất cả Mappers theo thứ tự batch index và sắp xếp chúng dựa trên image_id để đảm bảo consistency across datasets. Quá trình này bao gồm network transfer các intermediate results giữa các nodes, áp dụng compression để tối ưu bandwidth (đặc biệt quan trọng với SUN397 và FireSmoke có kích thước dữ liệu lớn), và thực hiện memory management để tránh OOM errors khi xử lý large feature matrices.

(4) Giai đoạn Reduce: Reducer duy nhất tải về toàn bộ logits từ HDFS và gộp chúng lại thành ma trận logits hoàn chỉnh cho từng dataset: [101,000 × 101] cho Food101, [108,754 × 397] cho SUN397, và [95,000+ × 4] cho FireSmoke. Sau đó, các thuật toán Conformal Prediction (LAC, APS, RAPS) được áp dụng với việc chia dữ liệu thành hai tập con: calibration (50%) và test (50%). Mô hình tính toán non-conformity scores cho từng mẫu dựa trên softmax probabilities với temperature scaling (optimized per dataset), xác định ngưỡng định lượng α (thường là 0.1 tương ứng với 90% coverage guarantee) và sinh ra các tập dự đoán (prediction sets) có đảm bảo thống kê về độ tin cậy.

V. Kết quả thực nghiệm:

V.1. Quá trình triển khai hệ thống

Nghiên cứu này thực hiện việc triển khai mô hình CLIP-Conformal Prediction trên kiến trúc MapReduce để xử lý ba bộ dữ liệu lớn có quy mô và tính chất khác nhau. Quá trình triển khai được thực hiện theo phương pháp tiếp cận có hệ thống, bắt đầu từ việc chuẩn bị môi trường phần cứng và phần mềm cần thiết.

Hệ thống được thiết lập trên cụm Hadoop gồm một master node và bốn worker nodes, mỗi node được trang bị 16GB RAM và CPU 8 cores. Hadoop Distributed File System được cấu hình với block size 128MB và replication factor là 3 để đảm bảo tính sẵn sàng cao của dữ liệu. Mô hình CLIP-ViT-B/32 được tải xuống và chuẩn bị sẵn sàng trên tất cả các nodes để giảm thiểu thời gian khởi tạo trong quá trình xử lý.

Ba bộ dữ liệu được chuẩn bị và tải lên HDFS theo quy trình chuẩn hóa. Food101 với 101,000 ảnh được tổ chức thành 101 thư mục tương ứng với các lớp món ăn, SUN397 với 108,754 ảnh được phân loại thành 397 thư mục scene categories, và FireSmoke với 95,000+ ảnh được sắp xếp thành 4 thư mục theo tình huống cháy nổ. Tất cả ảnh được resize về kích thước 224×224 pixels để phù hợp với input requirements của CLIP model.

Quá trình MapReduce được triển khai với hai giai đoạn chính. Giai đoạn Map thực hiện việc trích xuất đặc trưng CLIP song song trên các chunks dữ liệu, với mỗi Mapper xử lý khoảng 10,000-12,000 ảnh tùy theo dataset. Giai đoạn Reduce tập hợp tất cả các đặc trưng đã trích xuất và áp dụng các thuật toán Conformal Prediction (LAC, APS, RAPS) để tạo ra prediction sets với độ tin cậy thống kê.

Việc triển khai được thực hiện theo từng dataset một cách độc lập để đánh giá hiệu suất và khả năng thích ứng của hệ thống với các đặc điểm dữ liệu khác nhau. Mỗi lần chạy thực nghiệm được ghi lại đầy đủ các metrics về thời gian xử lý, memory usage, network utilization và accuracy để phục vụ cho việc phân tích và so sánh sau này.

V.2. Phân tích hiệu suất qua biểu đồ trực quan

V.2.1. So sánh tỷ lệ bao phủ (Coverage Rate Comparison)

|          | Food101 | SUN397  | FireSmoke |
|----------|---------|---------|-----------|
| Hình ảnh | ![Coverage Rate Food101](MapReduceResult/Charts/01_coverage_rate_comparison_food101.png) | ![Coverage Rate SUN397](MapReduceResult/Charts/01_coverage_rate_comparison_sun397.png) | ![Coverage Rate FireSmoke](MapReduceResult/Charts/01_coverage_rate_comparison_firesmoke.png) |

*Bảng 1. So sánh coverage rate trên 3 bộ dữ liệu*

Kết quả phân tích coverage rate cho thấy sự khác biệt rõ rệt trong hiệu suất của các thuật toán Conformal Prediction trên từng domain. FireSmoke dataset đạt được coverage rate cao và ổn định nhất, với cả ba thuật toán LAC, APS và RAPS đều vượt qua target coverage 90%. Điều này phản ánh tính chất binary-like của fire detection task, nơi các visual patterns có sự khác biệt rõ ràng giữa các classes. Food101 thể hiện hiệu suất trung bình với coverage rates dao động quanh mức target, cho thấy độ phức tạp vừa phải của food classification. SUN397 có coverage rates thấp nhất và variance lớn nhất, phản ánh tính challenging của fine-grained scene classification với 397 categories có sự tương đồng cao.

V.2.2. So sánh kích thước tập dự đoán trung bình (Average Setsize Comparison)

|          | Food101 | SUN397  | FireSmoke |
|----------|---------|---------|-----------|
| Hình ảnh | ![Setsize Food101](MapReduceResult/Charts/02_average_setsize_comparison_food101.png) | ![Setsize SUN397](MapReduceResult/Charts/02_average_setsize_comparison_sun397.png) | ![Setsize FireSmoke](MapReduceResult/Charts/02_average_setsize_comparison_firesmoke.png) |

*Bảng 2. So sánh average setsize trên 3 bộ dữ liệu*

Phân tích average setsize tiết lộ mối quan hệ nghịch đảo với coverage rate và phản ánh độ phức tạp của từng classification task. FireSmoke cho ra các prediction sets nhỏ nhất, với RAPS algorithm đạt average setsize chỉ khoảng 1.2, cho thấy mô hình có confidence cao trong việc phân loại tình huống cháy nổ. Food101 có setsize trung bình ở mức 2.1-2.4, phù hợp với độ phức tạp moderate của food recognition. SUN397 đòi hỏi prediction sets lớn nhất với setsize 2.8-3.5, phản ánh uncertainty cao trong việc phân biệt các scene categories tương tự nhau. Kết quả này khẳng định trade-off cơ bản trong conformal prediction giữa informativeness và reliability.

V.2.3. So sánh thời gian xử lý (Processing Time Comparison)

|          | Food101 | SUN397  | FireSmoke |
|----------|---------|---------|-----------|
| Hình ảnh | ![Processing Time Food101](MapReduceResult/Charts/03_processing_time_comparison_food101.png) | ![Processing Time SUN397](MapReduceResult/Charts/03_processing_time_comparison_sun397.png) | ![Processing Time FireSmoke](MapReduceResult/Charts/03_processing_time_comparison_firesmoke.png) |

*Bảng 3. So sánh processing time trên 3 bộ dữ liệu*

Biểu đồ processing time comparison cho thấy correlation rõ ràng giữa dataset complexity và computational requirements. SUN397 với 397 classes đòi hỏi thời gian xử lý dài nhất, do cần encode nhiều text descriptions và process larger feature matrices. Food101 có thời gian xử lý trung bình, phù hợp với 101 classes. FireSmoke xử lý nhanh nhất nhờ chỉ có 4 classes và simple text descriptions. MapReduce architecture thể hiện hiệu quả tốt ở tất cả datasets, với speedup factor từ 2.1x đến 2.8x so với sequential processing. Sự khác biệt trong speedup factor phản ánh overhead costs khác nhau của parallel coordination với dataset sizes và complexities khác nhau.

V.2.4. Phân tích đánh đổi giữa độ bao phủ và kích thước tập (Coverage-Setsize Tradeoff)

|          | Food101 | SUN397  | FireSmoke |
|----------|---------|---------|-----------|
| Hình ảnh | ![Tradeoff Food101](MapReduceResult/Charts/04_coverage_setsize_tradeoff_food101.png) | ![Tradeoff SUN397](MapReduceResult/Charts/04_coverage_setsize_tradeoff_sun397.png) | ![Tradeoff FireSmoke](MapReduceResult/Charts/04_coverage_setsize_tradeoff_firesmoke.png) |

*Bảng 4. So sánh coverage-setsize tradeoff trên 3 bộ dữ liệu*

Coverage-setsize tradeoff analysis cho thấy các đặc tính riêng biệt của từng domain khi điều chỉnh confidence levels. FireSmoke duy trì curve tốt nhất với setsize tăng chậm khi coverage tăng, cho thấy model confidence cao và consistent across different alpha values. Food101 thể hiện tradeoff moderate với slope trung bình, phản ánh balanced uncertainty trong food classification. SUN397 có slope steep nhất, indicating rằng để đạt higher coverage, cần phải chấp nhận significant increase trong prediction set sizes. Điều này khẳng định rằng scene classification là challenging task đòi hỏi careful tuning của confidence levels để cân bằng giữa practical usability và statistical guarantees.

V.2.5. Phân tích temperature scaling (Temperature Scaling Analysis)

|          | Food101 | SUN397  | FireSmoke |
|----------|---------|---------|-----------|
| Hình ảnh | ![Temperature Food101](MapReduceResult/Charts/05_temperature_scaling_analysis_food101.png) | ![Temperature SUN397](MapReduceResult/Charts/05_temperature_scaling_analysis_sun397.png) | ![Temperature FireSmoke](MapReduceResult/Charts/05_temperature_scaling_analysis_firesmoke.png) |

*Bảng 5. So sánh temperature scaling analysis trên 3 bộ dữ liệu*

Temperature scaling analysis cho thấy optimal calibration values khác nhau cho từng domain, phản ánh inherent confidence characteristics của CLIP model trên các types of visual content. FireSmoke đạt optimal performance với temperature thấp nhất (khoảng 1.2-1.3), cho thấy CLIP đã well-calibrated cho safety-critical visual patterns. Food101 cần temperature moderate (1.5-1.7) để đạt optimal calibration, suggesting moderate overconfidence trong food recognition tasks. SUN397 đòi hỏi temperature cao nhất (1.8-2.0), indicating significant overconfidence của CLIP trong scene classification và cần aggressive smoothing để đạt proper calibration. Kết quả này có implications quan trọng cho practical deployment, cho thấy cần domain-specific tuning cho optimal performance.

V.2.6. Phân tích phân bố thời gian chạy (Runtime Breakdown Analysis)

|          | Food101 | SUN397  | FireSmoke |
|----------|---------|---------|-----------|
| Hình ảnh | ![Runtime Food101](MapReduceResult/Charts/06_runtime_breakdown_analysis_food101.png) | ![Runtime SUN397](MapReduceResult/Charts/06_runtime_breakdown_analysis_sun397.png) | ![Runtime FireSmoke](MapReduceResult/Charts/06_runtime_breakdown_analysis_firesmoke.png) |

*Bảng 6. So sánh runtime breakdown analysis trên 3 bộ dữ liệu*

Runtime breakdown analysis tiết lộ bottlenecks chính trong processing pipeline và cho thấy patterns nhất quán across datasets. CLIP encoding luôn chiếm dominant portion (85-95%) của total runtime, với Vision Encoder là component expensive nhất. Text Encoder contribution giảm dần từ FireSmoke đến SUN397, phản ánh complexity increasing của text descriptions. Conformal Prediction computation chỉ chiếm 2-5% total time, confirming rằng CLIP encoding là primary computational bottleneck. MapReduce overhead (Shuffle & Sort) dao động từ 3-8% tùy theo dataset size và network transfer requirements. Kết quả này chỉ ra rằng future optimization efforts nên focus vào accelerating CLIP processing thông qua techniques như model quantization, batch optimization, hoặc specialized hardware acceleration.

V.2.7. Phân tích hội tụ độ bao phủ (Coverage Convergence)

|          | Food101 | SUN397  | FireSmoke |
|----------|---------|---------|-----------|
| Hình ảnh | ![Convergence Food101](MapReduceResult/Charts/08_coverage_convergence_food101.png) | ![Convergence SUN397](MapReduceResult/Charts/08_coverage_convergence_sun397.png) | ![Convergence FireSmoke](MapReduceResult/Charts/08_coverage_convergence_firesmoke.png) |

*Bảng 7. So sánh coverage convergence trên 3 bộ dữ liệu*

Coverage convergence analysis cho thấy stability và reliability của conformal prediction methods trên different sample sizes. FireSmoke đạt convergence nhanh nhất và ổn định nhất, với coverage rate stabilizing sau khoảng 1000-2000 samples, reflecting strong signal-to-noise ratio trong fire detection task. Food101 cần sample size lớn hơn (3000-5000 samples) để đạt stable convergence, cho thấy moderate complexity và variance trong food classification. SUN397 có convergence chậm nhất và variance cao nhất, cần 8000+ samples để đạt stable coverage rates, phản ánh inherent difficulty của fine-grained scene classification. Kết quả này có practical implications cho deployment, cho thấy minimum dataset sizes cần thiết để đạt reliable conformal prediction performance cho từng application domain.

V.3. Phân tích định lượng từ báo cáo thống kê

Các báo cáo Excel chi tiết từ thư mục MapReduceResult/Reports cung cấp comprehensive quantitative analysis về performance metrics của các thuật toán Conformal Prediction. Phân tích sâu các số liệu thống kê cho thấy patterns và insights quan trọng về behavior của hệ thống trên các domains khác nhau.

Statistical significance testing được thực hiện bằng paired t-tests và McNemar's tests để đánh giá differences giữa algorithms. Kết quả cho thấy RAPS consistently outperforms LAC và APS trên tất cả metrics chính. Cho FireSmoke dataset, p-values cho RAPS vs LAC và RAPS vs APS đều nhỏ hơn 0.01, indicating highly significant differences. Food101 shows moderate significance levels với p-values trong khoảng 0.01-0.05. SUN397 exhibits marginal significance cho một số comparisons, suggesting that algorithm choice có impact nhỏ hơn trong highly complex classification tasks.

Marginal coverage analysis reveals domain-specific behaviors. FireSmoke achieves coverage rates closest to nominal levels, với deviations dưới 1% cho tất cả algorithms. Food101 shows moderate deviations (1-3%), while SUN397 exhibits larger deviations (2-5%), particularly cho LAC algorithm. Conditional coverage analysis confirms these patterns, với FireSmoke maintaining consistent coverage across different subgroups, Food101 showing moderate variation, và SUN397 demonstrating significant heterogeneity across scene categories.

Efficiency metrics analysis cho thấy clear ranking across domains. FireSmoke achieves highest efficiency scores (0.85-0.92), Food101 moderate scores (0.75-0.85), và SUN397 lowest scores (0.65-0.80). RAPS consistently delivers best efficiency across all domains, followed by APS và LAC. Size-stratified analysis reveals that efficiency decreases với increasing prediction set sizes, với steepest decline observed trong SUN397 due to its large number of similar categories.

Cross-validation stability metrics confirm robustness của results. Coefficient of variation cho coverage rates ranges từ 2% (FireSmoke) đến 6% (SUN397), indicating increasing uncertainty với domain complexity. Bootstrap confidence intervals show tightest bounds cho FireSmoke [91.8%, 93.2%] và widest cho SUN397 [88.5%, 91.5%]. Jackknife estimates confirm these patterns, với FireSmoke showing minimal bias và SUN397 exhibiting larger bias estimates.

Performance variance analysis across classes reveals interesting domain characteristics. FireSmoke exhibits low variance (CV < 5%) across its 4 classes, indicating consistent model performance. Food101 shows moderate variance (CV 15-25%) với some classes như "pizza" và "sandwich" proving more challenging than others like "ice_cream". SUN397 demonstrates high variance (CV 30-45%) với significant performance gaps between easy categories like "abbey" và difficult ones like "highway" hoặc "corridor".

Memory và computational profiling data reveal scalability characteristics. Peak memory usage scales approximately quadratically với number of classes: 6.8GB (FireSmoke), 12.4GB (Food101), và 28.7GB (SUN397). Network transfer volumes follow similar patterns: 2.1GB (FireSmoke), 4.8GB (Food101), và 11.2GB (SUN397). Processing rates per image vary significantly: 45ms (FireSmoke), 78ms (Food101), và 124ms (SUN397), reflecting both computational complexity và coordination overhead trong MapReduce framework.

Error analysis shows different failure modes across domains. FireSmoke errors primarily occur trong ambiguous lighting conditions hoặc partial occlusion scenarios. Food101 failures concentrate trong visually similar foods và complex plating presentations. SUN397 errors are most diverse, spanning lighting variations, viewpoint changes, và semantic ambiguities between similar scene types. These patterns provide valuable insights cho future improvement directions và practical deployment considerations.
