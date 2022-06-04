import numpy as np
import json
import pickle
from scipy.spatial import distance as dist
from collect_input import get_landmarks, get_distances
from collect_input import connectivity
from scipy.optimize import least_squares

import cv2
import tensorflow as tf
import tensorflow_addons as tfa

## read training data
def read_data():
    full_data = []
    with open('./new_train.jsonl') as f:
        for line in f:
            full_data.append(json.loads(line))
    return full_data

def sort_by_weight(distance, data):
    # calculate the beauty weight of each data
    for d in data:
        normalized_distance = np.multiply(np.array(d['distances']), 1 / d['area_sqrt'])
        weight = d['score'] / np.linalg.norm(distance - normalized_distance)
        d['normalized_distance'] = normalized_distance
        d['weight'] = weight
    # sort data by the beauty weight
    sorted_data = sorted(data, key=lambda d: d['weight'], reverse=True)
    # return weights and weighted distances
    weights = []
    distances = []
    for d in sorted_data:
        w = d['weight']
        weights.append(w)
        distances.append(d['normalized_distance'])
    return weights, distances

def knn(weights, distances, k, beauty_estimator):
    vectors = np.array(distances[:k])
    w = np.array(weights[:k])
    total_weight = np.sum(w)
    product = np.dot(w, vectors)
    weighted_sum = np.multiply(product, 1 / total_weight)
    weighted_sum = np.reshape(weighted_sum,(1, -1))
    score = beauty_estimator.predict(weighted_sum)
    return score, weighted_sum

# def get_feature(landmark_indice):
#     if landmark_indice <= 16:
#         return "jaw"
#     elif landmark_indice <= 21:
#         return "left eyebrow"
#     elif landmark_indice <= 26:
#         return "right eyebrow"
#     elif landmark_indice <= 30:
#         return "nose bridge"
#     elif landmark_indice <= 35:
#         return "lower nose"
#     elif landmark_indice <= 41:
#         return "left eye"
#     elif landmark_indice <= 47:
#         return "right eye"
#     elif landmark_indice <= 59:
#         return "outer lip"
#     elif landmark_indice <= 67:
#         return "inner lip"
def get_feature(landmark_indice):
    if landmark_indice <= 16:
        return "jaw"
    elif landmark_indice <= 21:
        return "left eyebrow"
    elif landmark_indice <= 26:
        return "right eyebrow"
    elif landmark_indice <= 30:
        return "nose bridge"
    elif landmark_indice <= 35:
        return "lower nose"
    # elif landmark_indice <= 41:
    #     return "left eye"
    elif landmark_indice <= 47:
        return "eyes"
    else:
        return "lip"

def model(landmarks, connectivity):
    model = []
    for edge in connectivity:
        x1 = landmarks[2 * edge[0]]
        y1 = landmarks[2 * edge[0] + 1]
        x2 = landmarks[2 * edge[1]]
        y2 = landmarks[2 * edge[1] + 1]
        model.extend([np.square(x1 - x2) + np.square(y1 - y2)])
    # (175,)
    return np.array(model)
    
def fun(landmarks, square_distances, connectivity, alpha):
    loss = (model(landmarks, connectivity) - square_distances)
    new_loss = []
    for i, l in enumerate(loss):
        new_loss.extend([alpha[i] * l])
    return np.array(new_loss)

def get_target_landmarks(original_landmarks, square_distances, connectivity):
    alpha = []
    for edge in connectivity:
        if get_feature(edge[0]) == get_feature(edge[1]):
            ## 10 or np.sqrt(10)
            alpha.extend([10])
        else: 
            alpha.extend([1])
    target_landmarks = least_squares(fun, x0=original_landmarks , args=(square_distances, connectivity, alpha),bounds=(0,360))
    # target_landmarks = least_squares(fun, x0=original_landmarks , args=(square_distances, connectivity, alpha), method='lm')
    return target_landmarks.x

def warp(img, original_landmarks, target_landmarks):
    img = img.astype(np.float32)
    original_landmarks = original_landmarks.astype(np.float32)
    target_landmarks = target_landmarks.astype(np.float32)

    img = img[np.newaxis, :, :, :]
    original_landmarks = original_landmarks[np.newaxis, :, :]
    target_landmarks = target_landmarks[np.newaxis, :, :]

    warped_img, flow_field = tfa.image.sparse_image_warp(img, original_landmarks, target_landmarks)
    return warped_img

def draw_delaunay(img, landmarks, connectivity):
    img = cv2.imread(img)
    delaunay_color = (255, 255, 255)
    for edge in connectivity:
        cv2.line(img, landmarks[edge[0]], landmarks[edge[1]], delaunay_color, 1, cv2.LINE_AA, 0)
    cv2.imwrite("./output/delaunay.jpg", img)

def main():
    # Given an image
    img = './frontal_faces/2a.jpg'

    # get the image's data
    original_landmarks, face_area_sqrt = get_landmarks(img)
    distance = get_distances(original_landmarks, connectivity)
    normalized_distance = np.multiply(np.array(distance), 1 / face_area_sqrt)

    draw_delaunay(img, original_landmarks, connectivity)
    # read train data
    data = read_data()
    weights, distances = sort_by_weight(normalized_distance, data)
    # load the trained beauty score estimator
    filename = './beauty_score_model.sav'
    beauty_estimator = pickle.load(open(filename, 'rb'))
    print("Initial score: ", beauty_estimator.predict(np.reshape(normalized_distance,(1, -1))))

    ### searching for the value of K that maximizes the SVR beauty score of the weighted sum
    # k = 1,.....15
    best_k = -1
    best_score = 0
    best_distance_vector = None
    ## KNN BEAUTIFICATION
    for k in range(1, 16):
        score, weighted_sum = knn(weights, distances, k, beauty_estimator)
        print(k, score)
        if score > best_score:
            best_score = score
            best_k = k
            best_distance_vector = weighted_sum
    print("k=",best_k)
    print("Beautified score: ", best_score)

    # k=15
    # score, weighted_sum = knn(weights, distances, k, beauty_estimator)
    # if score > best_score:
    #     best_score = score
    #     best_k = k
    #     best_distance_vector = weighted_sum
    # print("k=",best_k)
    # print("Beautified score: ", best_score)

    # Get target landmarks from the beautified distance vector
    flat_landmark = np.array(original_landmarks).flatten()
    square_distance = [np.square(d * face_area_sqrt) for d in best_distance_vector]
    square_distance = np.array(square_distance).flatten()

    target_landmarks = get_target_landmarks(flat_landmark, square_distance, connectivity)
    
    original_landmarks = np.array(original_landmarks)
    target_landmarks = np.reshape(target_landmarks, (-1, 2))
    print(original_landmarks)
    print(target_landmarks)
    img = cv2.imread(img)
    warped_img = warp(img, original_landmarks, target_landmarks)
    warped_img = np.array(warped_img)
    warped_img = np.squeeze(warped_img)

    cv2.imwrite("./output/input.jpg", img)
    cv2.imwrite("./output/result.jpg", warped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":  
    main()