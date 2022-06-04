import dlib, cv2
import numpy as np
import os, os.path
from scipy.spatial import distance as dist
import json

valid = [1, 4, 7, 8, 10, 14, 16, 22, 23, 26, 28, 34, 41, 43, 46, 48, 49, 55, 57, 61, 62, 66, 67, 69, 73, 74, 75, 79, 84, 86, 87, 88, 92, 94, 95, 98, 99, 100, 106, 107, 109, 110, 114, 115, 121, 129, 131, 132, 134, 135, 141, 145, 150, 157, 160, 161, 165, 167, 168, 169, 171, 173, 176, 177, 178]
connectivity = [[0, 1], [0, 17], [0, 36], [1, 2], [1, 31], [1, 36], [1, 41], [2, 3], [2, 31], [2, 48], [3, 4], [3, 48], [4, 5], [4, 48], [5, 6], [5, 48], [6, 7], [6, 48], [6, 59], [7, 8], [7, 58], [7, 59], [8, 9], [8, 57], [8, 58], [9, 10], [9, 55], [9, 56], [9, 57], [10, 11], [10, 54], [10, 55], [11, 12], [11, 54], [12, 13], [12, 54], [13, 14], [13, 54], [14, 15], [14, 35], [14, 54], [15, 16], [15, 35], [15, 45], [15, 46], [16, 26], [16, 45], [17, 18], [17, 36], [18, 19], [18, 36], [18, 37], [19, 20], [19, 24], [19, 37], [19, 38], [20, 21], [20, 23], [20, 24], [20, 38], [20, 39], [21, 22], [21, 23], [21, 27], [21, 39], [22, 23], [22, 27], [22, 42], [22, 43], [23, 24], [23, 43], [23, 44], [24, 25], [24, 44], [25, 26], [25, 44], [25, 45], [26, 45], [27, 28], [27, 39], [27, 42], [28, 29], [28, 39], [28, 42], [29, 30], [29, 31], [29, 35], [29, 39], [29, 42], [29, 47], [30, 31], [30, 32], [30, 33], [30, 34], [30, 35], [31, 32], [31, 39], [31, 40], [31, 41], [31, 48], [31, 49], [31, 50], [32, 33], [32, 50], [33, 34], [33, 50], [33, 51], [33, 52], [34, 35], [34, 52], [34, 53], [35, 46], [35, 47], [35, 53], [35, 54], [36, 37], [36, 41], [37, 38], [37, 40], [37, 41], [38, 39], [38, 40], [39, 40], [40, 41], [42, 43], [42, 47], [43, 44], [43, 47], [44, 45], [44, 46], [44, 47], [45, 46], [46, 47], [48, 49], [48, 59], [48, 60], [49, 50], [49, 59], [49, 60], [49, 61], [50, 51], [50, 61], [51, 52], [51, 61], [51, 62], [51, 63], [51, 66], [51, 67], [52, 53], [52, 63], [52, 65], [53, 54], [53, 55], [53, 64], [53, 65], [54, 55], [54, 64], [55, 56], [55, 64], [55, 65], [56, 57], [56, 65], [56, 66], [57, 58], [57, 66], [58, 59], [58, 61], [58, 62], [58, 66], [58, 67], [59, 60], [59, 61], [61, 67], [62, 66], [62, 67], [63, 65], [63, 66], [65, 66]]

def get_landmarks(file_name):
    Model_PATH = "dlib-models/shape_predictor_68_face_landmarks.dat"
    frontalFaceDetector = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

    img = cv2.imread(file_name)
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = []
    face = frontalFaceDetector(imageRGB, 0)
    faceRectangleDlib = dlib.rectangle(face[0])
    face_area_sqrt = np.sqrt(faceRectangleDlib.width() * faceRectangleDlib.height())
    landmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
    landmarks_list = []
    for i in range(0, landmarks.num_parts):
        landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

    return landmarks_list, face_area_sqrt

def get_distances(landmarks, connectivity):
    distances = []
    for edge in connectivity:
        pt1 = edge[0]
        pt2 = edge[1]
        d = dist.euclidean(landmarks[pt1], landmarks[pt2])
        distances.append(d)

    return distances

def collect_data():
    scores = []
    images = []

    for v in valid:
        file_name = f"./frontal_faces/{v}a.jpg"
        images.append(file_name)

    score1 = []
    with open('./score/score1.txt') as f:
        lines = f.readlines()
        for v in valid:
            score1.append(float(lines[v - 1]))

    score2 = []
    with open('./score/score2.txt') as f:
        lines = f.readlines()
        for v in valid:
            score2.append(float(lines[v - 1]))

    score3 = []
    with open('./score/score3.txt') as f:
        lines = f.readlines()
        for v in valid:
            score3.append(float(lines[v - 1]))
    score4 = []
    with open('./score/score4.txt') as f:
        lines = f.readlines()
        for v in valid:
            score4.append(float(lines[v - 1]))

    scores = [(s1 + s2 + s3 + s4)/4 for s1, s2, s3, s4 in zip(score1, score2, score3, score4)]
    return images, scores

def main():
    output_file = './new_train.jsonl'
    images, scores = collect_data()
    with open(os.path.join(output_file), "w") as f1:
        for i, (img, score) in enumerate(zip(images, scores)):
            print(i)
            landmarks, face_area_sqrt = get_landmarks(img)
            distances = get_distances(landmarks, connectivity)

            ## write to output file
            json_string = {
                "file_name": img,
                "score": score,
                "landmarks": landmarks,
                "distances": distances,
                "area_sqrt": face_area_sqrt
            }

            json.dump(json_string, f1)
            f1.write('\n')

if __name__ == "__main__":
    main()
