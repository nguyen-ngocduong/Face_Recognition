from utils.detectFace import detect
from utils.ExtractEmbedding import *
from utils.face_regconition import *
import cv2

def process_image_for_recognition(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
def detect_and_recognization(frame, detect_model, facenet_model, l2_normalize, embedding_df):
    """
    Phat hien va nhan dien khuon mat trong anh
    """
    original_frame = frame.copy()
    _, bounding_box_list = detect(detect_model, frame)
    for box in bounding_box_list:
        x1, y1, x2, y2 = box
         # Thêm padding để crop tốt hơn
        h, w = frame.shape[:2]
        padding = 10
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding) 
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        # ve bounding box
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        img_crop = original_frame[(y1):(y2), (x1):(x2)]
        if img_crop.size == 0:
            continue
        #Xu ly anh
        process_image = process_image_for_recognition(img_crop)
        # trich xuat dac trung
        vector_embedding = extract_feature(facenet_model, process_image)
        vector_embedding = l2_normalize.transform(vector_embedding.reshape(1, -1))[0]
        person_name, do_tin_cay = cal_similar(vector_embedding, embedding_df)
        #Hien thi
        label = f"{person_name} ({do_tin_cay:.2f})"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame