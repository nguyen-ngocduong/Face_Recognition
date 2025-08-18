import numpy as np
from numpy.linalg import norm

def convert_str2array(input_str):
    number_list = [float(num) for num in input_str.split()]
    array = np.array(number_list, dtype = np.float32)
    return array

def cosine(v1, v2):
    return 1 - np.dot(v1, v2) / (norm(v1) * norm(v2))

def cal_similar(vectorC, embedding_df):
    """
    vectorC: numpy array (embedding khuôn mặt mới)
    embedding_df: DataFrame chứa ['labels', 'embeddings']
    """
    distances = []
    
    # duyệt qua từng embedding trong database
    for row in range(embedding_df.shape[0]):
        emb = convert_str2array(embedding_df.iloc[row, 1])   # embeddings nằm ở cột 1
        dis = cosine(emb, vectorC)  # khoảng cách cosine (0 = giống hệt, càng nhỏ càng giống)
        distances.append(dis)
    
    min_distance_index = np.argmin(distances)  # lấy chỉ số khoảng cách nhỏ nhất
    
    if distances[min_distance_index] > 0.3:  # ngưỡng 0.3 (tùy chỉnh)
        person_name = "Khong phat hien guong mat"
    else:
        person_name = embedding_df.iloc[min_distance_index, 0]  # label nằm ở cột 0
    
    return person_name
