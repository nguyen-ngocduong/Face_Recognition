# FACE DETECTION

## Hiểu về face detection

- *Face detection* là quá trình tự động định vị và xác định khuôn mặt người trong hình ảnh kỹ thuật hay trong một video.

## Tính năng phát hiện khuôn mặt hoạt động như thế nào?

1. Tiền xử lý ảnh

- Điều chỉnh độ sáng hoặc cân bằng màu để cải thiệnchất lượng hình ảnh
- Có thể chuyểnđổi hình ảnh sang dạng thang độ xám vì các đặc điểmtrên khuôn mặt thường dễ phát hiện hơn trong hình ảnhthang độ xám

2. Trích xuất tính năng

- Các thuật toán phân tích hình ảnh để trích xuất cácđặc điểm biểu thị đặc điểm khuôn mặt. Các đặcđiểm này có thể bao gồm các cạnh, dải màu, hoa vănkết cấu hoặc các điểm mốc cụ thể trên khuôn mặt.
- Các kỹ thuật trích xuất đặc điểm nhằm mục đíchnắm bắt những đặc điểm riêng biệt của khuôn mặtcon người để phân biệt chúng với các vật thể hoặcyếu tố nền khác trong hình ảnh.

3. Phân loại

- Sử dụng các mô hình Machine Learning hoặc Deep Learning đểphân loại các vùng có chứa khuôn mặt hay không.

4. Trực quan hóa

- Khi một vùng được xác định là có khuôn mặt thì nósẽ xác định vị trí và kích thước của khuôn mặt cótrong ảnh hoặc video.
- Thường được biểu diễn dưới dạng hộp - **BoudingBox** => cho biết vị trí và phạm vi có chứa khuôn mặttrong ảnh.
- Cácthuật toán định vị có thể sử dụng các kỹ thuậtnhư khớp mẫu, phương pháp cửa sổ trượt hoặc mạngnơ-ron tích chập (CNN) để xác định chính xác khuôn mặttrong hình ảnh.

## Phương pháp mạng nơ-ron trong Face Detection:


1. Mạng nơ-ron tích chập (CNN) : CNN đãrất thành công trong các nhiệm vụ phát hiện khuôn mặt.Các kiến trúc như R-CNN, Fast R-CNN và Faster R-CNN đã đượcsử dụng cho mục đích này.
2. YOLO : YOLO là một thuật toán phát hiện đối tượng phổ biến khác có thể được sử dụng để nhận diện khuôn mặt. Thuật toán này nổi tiếng với hiệu suất theo thời gian thực.
3. MTCNN (Mạng tích chập xếp tầng đa nhiệm) : MTCNN là một mô hình mạng nơ-ron được thiết kế	chuyên biệt cho các tác vụ phát hiện khuôn mặt. Nó phát hiện khuôn mặt ở nhiều tỷ lệ và góc độ khác	nhau chỉ trong một lần truyền.
4. MobileNet : MobileNet là một kiến trúc mạng nơ-ron nhẹ thường được sử dụng để phát hiện khuôn mặt trên các thiết bị di động do tính hiệu quả của nó.

## Chỉ số hiệu suất trên gương mặt:

- *Accuracy* : Tỷ lệ phần trăm gương mặt được phát hiện chính xáctrong các tập dữ liệu
- *Precision* : Độ chính xác đo lường tỷ lệ khuôn mặt được phát hiện chính xác trong số tất cả các trường hợp được phát hiện là khuôn mặt

  $$
  Precision = \frac{\text{Số kết quả dương tính thật}}{\text{Số kết quả dương tính thật} + \text{Số kết quả dương tính giả}}
  $$
- *Recall* : được gọi là độ nhạy hoặc tỷ lệ dương tính thật, đo lường tỷ lệ khuôn mặt được phát hiện chính xác trong số tất cả các khuôn mặt thực tế trong tập dữ liệu. Chỉ số này cho biết khả năng của thuật toán trong việc xác định tất cả các trường hợp dương tính.

  $$
  Recall = \frac{\text{Số kết quả dương tính thật}}{\text{Số dương tính thật + Số âm tính giả}}
  $$
- *F1 Score* : Điểm F1 là giá trị trung bình hài hòa của *Precision* và *Recall*.

  $$
  \text{F1 Score} = \frac{\text{2 * (Precision * Recall) }}{\text{Precision + Recall}}
  $$
- ***Tỷ lệ Dương tính Giả (FPR)*** : FPR đo lường tỷ lệ các khuôn mặt không phải khuôn mặt được phân loại sai thành khuôn mặt.
- ***Tỷ lệ Âm tính Giả (FNR)*** : FNR đo lường tỷ lệ khuôn mặt thực tế bị phân loại sai là không phải khuôn mặt
