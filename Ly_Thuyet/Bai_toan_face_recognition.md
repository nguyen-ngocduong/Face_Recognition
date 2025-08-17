# **Face Recognition** - Nhận dạng khuôn mặt

Face Recognition là bài toán nhận dạng và xác thực người dựa vào khuôn mặt của ho. Phuong phap pho bien nhat la phuong phap FaceNet (A Unified Embedding for Face Recognition and Clustering)

## Các khái niệm cơ bản:

1. *Embedding Vector* : Là một vector với dimension cố định (thường có chiều nhỏ hơn các Feature Vector bình thường), đã được học trong quá trình train và đại diện cho một tập các feature có trách nhiệm trong việc phân loại các đối tượng trong chiều không gian đã được biến đổi.
2. ****Inception V1**: ột cấu trúc mạng CNN được giới thiệu vào năm 2014 của Google, với đặc trưng là các khối Inception. Khổi này cho phép mạng được học theo cấu trúc song song, nghĩa là với 1 đầu vào có thể được đưa vào nhiều các lớp Convolution khác nhau để đưa ra các kết quả khác nhau, sau đó sẽ được Concatenate vào thành 1 output.**

![Alt text](/home/nguyenngocduong/Desktop/face_recognition_project/Ly_Thuyet/jpg/anh2.webp)

3. **Triplet Loss**: là một hàm mất mát dùng để huấn luyện mạng nơ-ron sao cho các khuôn mặt cùng người có khoảng cách gần, còn khuôn mặt khác người có khoảng cách xa.

$$
L = \sum_{i} \left[ \left\| f(x_{i}^{a}) - f(x_{i}^{p}) \right\|_{2}^{2} - \left\| f(x_{i}^{a}) - f(x_{i}^{n}) \right\|_2^2 + \alpha \right]_+
$$

- Triplet Loss đưa ra một công thức mới bao gồm 3 giá trị đầu vào gồm **anchor** (${x_{i}^a}$): ảnh đầu ra của mạng, **positive** ($x_{i}^{p}$) : ảnh cùng là 1 người với anchor và *negative ($x_{i}^n$) : ảnh không cùng là 1 người với anchor. $\alpha$ là margin (lề thêm) giữa cặp positive với negative, độ sai lệch cần thiết tối thiểu giữa 2 miền giá trị.

---

## Luồng xử lý của bài toán Face Recognition

Bài toán Face Recognition bắt buộc phải bao gồm tối thiếu 3 bước sau:

- Bước 1: **Face Detection** - Xác định vị trí của khuôn mặt trong ảnh (hoặc video frame).Vùng này sẽ được đánh dấu bằng một hình chữ nhật bao quanh.
- Bước 2: **Face Extraction** (Face Embedding)- Trích xuất đặc trưng của khuôn mặt thành một vector đặc trưng trong không gian nhiều chiểu (thường là 128 chiều).
- Bước 3: **Face Classification** (Face Authentication - Face Verification - Face Identification).

Ngoài 3 bước trên, trong thực tế chúng ta thường bổ sung thêmmột số bước để tăng độ chính xác nhận diện:

- **Image Preprocessing** : Xử lý giảm nhiễu, giảm mờ, giảm kích thước, chuyển sang ảnh xám, chuẩn hóa, …
- *Face Aligment* : Nếu ảnh khuôn mặt bị nghiêng thì căn chỉnhlại sao cho ngay ngắn.
- Kết hợp nhiều phương pháp khác nhau tại bước 3.

![Alt text](/home/nguyenngocduong/Desktop/face_recognition_project/Ly_Thuyet/jpg/anh4.png)
