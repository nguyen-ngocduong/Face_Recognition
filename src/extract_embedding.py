import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

def extract_embedding(image_array, model):
    """
    Trích xuất embedding từ ảnh sử dụng InceptionResNetV2.
    """
    embeddings = []
    for img in image_array:
        img_resized = tf.image.resize(img, (299, 299))
        img_resized = tf.expand_dims(img_resized, axis=0)  # thêm batch dim
        img_preprocessed = preprocess_input(img_resized)
        embedding = model(img_preprocessed, training=False)
        embedding = embedding.numpy()[0]  # lấy phần tử đầu tiên, shape (1536,)
        embeddings.append(embedding)
    return np.array(embeddings)

def find_best_match(new_embedding, embeddings, labels, threshold):
    """
    Tìm kiếm embedding gần nhất với một ngưỡng nhất định.
    """
    distances = np.linalg.norm(embeddings - new_embedding, axis=1) # tinh khoang cach Euclid
    min_idx = np.argmin(distances)
    if distances[min_idx] < threshold:
        return labels[min_idx], distances[min_idx]
    else:
        return "Unknown", distances[min_idx]