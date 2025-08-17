# MTCNN - Multi-task Cascaded Convolutional Networks

## 1. Tìm hiểu về MTCNN

- MTCNN là một **bộ ba mạng CNN xếp tầng** để **phát hiện khuôn mặt**

![Alt text](/home/nguyenngocduong/Desktop/face_recognition_project/Ly_Thuyet/jpg/anh1.webp)

- **P-Net (Proposal Network)**: quét nhanh trên **tháp ảnh (image pyramid)** để đưa ra rất nhiều “ô đề cử” (candidate boxes) + độ tin cậy.
- R-Net(Refine Network): lọc bớt các ô đề cử “nhiễu” và **hiệu chỉnh (regress) hộp** chính xác hơn.
- **O-Net (Output Network)**: xác nhận lần cuối, **dự đoán 5 landmarks** (hai mắt, mũi, khóe miệng trái/phải) và tinh chỉnh hộp thêm một lần nữa.
- Mỗi tầng gồm 3 đầu ra: \*face classification(Nhị phân)\*, \*Bouding Box Regression(Chỉnh lại hộp)\*\, \***Landmark regression** (chỉ O-Net)\*\.

---



## 2. Hàm mất mát (loss) & nhãn mẫu (multi-task)

- Mỗi mạng (đặc biệt O-Net) tối ưu tổng hợp:

$$
L = L_{\text{cls}} + \lambda \cdot L_{\text{box}} + \lambda_{\text{lm}} \cdot L_{\text{lm}}
$$

- $L_{\text{cls}}$ : **cross-entropy** nhi phan
- $L_{\text{box}}$: L2 (hoặc Smooth-L1) cho offsets  [dx1,dy1,dx2,dy2].
- $L_{\text{lm}}$: **L2** cho 10 toạ độ landmarks

---



## 3. Quy trình huấn luyện (Training Pipeline)

- **Dữ liệu đầu vào**: Ảnh khuôn mặt với nhãn boundingbox và landmark.
- **Huấn luyện từng mạng**: P-Net → R-Net → O-Net, mỗi mạng được huấn luyện riêng biệt với các nhãn phù hợp (face/non-face, bounding box, landmark).
- **Kỹ thuật hard negative mining**: Lấy các vùng không phải khuôn mặt nhưng bị nhận nhầm để huấn luyện lại, giúp tăng độ chính xác.

---

## 4. Pipelinephát hiện khuôn mặt (Detection Pipeline)


1. **P-Net**: Quét ảnh ở nhiều tỷ lệ (image pyramid), sinh racác candidate boxes.

2. **Non-Maximum Suppression (NMS)**: Loại bỏ các hộp trùng lặp,giữ lại hộp có độ tin cậy cao nhất.

3. **R-Net**: Nhận các candidate boxes từ P-Net, lọc và hiệuchỉnh lại hộp.

4. **NMS lần 2**: Tiếp tục loại bỏ trùng lặp.

5. **O-Net**: Dự đoán landmark và tinh chỉnh hộp cuối cùng.
6. **NMS lần cuối**: Đưa ra kết quả cuối cùng.

---
