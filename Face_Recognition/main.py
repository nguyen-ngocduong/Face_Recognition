from utils.detectFace import *
from utils.ExtractEmbedding import *
from utils.face_regconition import *
import cv2
from ultralytics import YOLO
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.preprocessing import Normalizer
from detect_and_regconition import *

detect_model = YOLO("Models/yolov11n-face.pt")
facenet_model = InceptionResNetV2(weights="imagenet", include_top=False, pooling='avg')
embedding_df = pd.read_csv("output/embeddings.csv")
l2_normalize = Normalizer('l2')

cap = cv2.VideoCapture(0)  # Mở camera
if not cap.isOpened():
    print("Khong thể mở camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera")
        break
    
    # Phát hiện và nhận diện khuôn mặt
    frame = detect_and_recognization(frame, detect_model, facenet_model, l2_normalize, embedding_df)
    
    cv2.imshow("Face Recognition", frame)
    # nhan q de thoat
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()