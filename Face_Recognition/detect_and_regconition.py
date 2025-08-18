from utils.detectFace import detect
from utils.ExtractEmbedding import *
from utils.face_regconition import *
import cv2

def process_image_for_recognition(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
def detect_and_recognization(frame, detect_model, facenet_model, l2_normalize, embedding_df):
    """
    Phat hien va nhan dien khuon mat trong anh
    """
    _, bounding_box_list = detect(detect_model, frame)
    for box in bounding_box_list:
        x1, y1, x2, y2 = box
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        img_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if img_crop.size == 0:
            continue
        #img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img_crop = process_image_for_recognition(img_crop)
        #img_crop = cv2.resize(img_crop, (160, 160))
        vector_embedding = extract_feature(facenet_model, img_crop)
        vector_embedding = l2_normalize.transform(vector_embedding.reshape(1, -1))[0]
        person_name = cal_similar(vector_embedding, embedding_df)
        cv2.putText(frame, person_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame