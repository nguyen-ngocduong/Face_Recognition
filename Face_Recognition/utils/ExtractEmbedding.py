import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
from tqdm import tqdm

embedding_model = InceptionResNetV2(weights = "imagenet", include_top = False, pooling = 'avg')

def process_image(image_path, target_size = (160, 160)):
    """
    Đọc ảnh và chuẩn hoá theo InceptionResNetV2
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image
def extract_feature(model, img):
    """
    Trích xuất embedding từ ảnh đã chuẩn hóa (chuẩn hóa trước khi truyền vào hàm này).
    img: numpy array, shape (160, 160, 3) hoặc (1, 160, 160, 3)
    model: keras model nhận diện khuôn mặt
    """
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    emb = model.predict(img, verbose=0)
    return emb.flatten()
def extract_embedding(input_csv, output_path, model=embedding_model):
    """
    Trich xuat embedding từ ảnh trong file CSV
    """
    df = pd.read_csv(input_csv)
    labels, embeddings = [], []
    print(f"Processing {len(df)} detected faces...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row['image']
        label = row['person']
        image = process_image(image_path)
        if image is None:
            continue

        emb = model.predict(image, verbose=0)
        emb_flat = emb.flatten()

        labels.append(label)
        # Chuyển vector thành chuỗi để lưu CSV
        embeddings.append(" ".join(map(str, emb_flat.tolist())))
        #Tao dataframe
        df_embeddings = pd.DataFrame({'person': labels, 'embedding': embeddings})
        #luu file csv
        df_embeddings.to_csv(output_path, index = False)
        print(f"[DONE] Saved embeddings to {output_path}")

if __name__ == "__main__":
    input_csv = "output/bounding_boxes.csv"
    output_path = "output/embeddings.csv"
    extract_embedding(input_csv, output_path, model= embedding_model)