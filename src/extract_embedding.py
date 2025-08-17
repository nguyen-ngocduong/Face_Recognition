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
