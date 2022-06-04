import dlib, cv2
import numpy as np 

def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False):
  points = []
  for i in range(startpoint, endpoint+1):
    point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
    points.append(point)

  points = np.array(points, dtype=np.int32)
  cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

def get_landmarks(img):
    Model_PATH = "dlib-models/shape_predictor_68_face_landmarks.dat"
    frontalFaceDetector = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = []
    face = frontalFaceDetector(imageRGB, 0)
    faceRectangleDlib = dlib.rectangle(face[0])
    landmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
    landmarks_list = []
    assert(landmarks.num_parts == 68)
    for i in range(0, landmarks.num_parts):
        landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

    return landmarks_list
# Use this function for 70-points facial landmark detector model
# we are checking if points are exactly equal to 68, then we draw all those points on face one by one
# def facePoints(image, faceLandmarks):
#     assert(faceLandmarks.num_parts == 68)
#     drawPoints(image, faceLandmarks, 0, 16)           # Jaw line
#     drawPoints(image, faceLandmarks, 17, 21)          # Left eyebrow
#     drawPoints(image, faceLandmarks, 22, 26)          # Right eyebrow
#     drawPoints(image, faceLandmarks, 27, 30)          # Nose bridge
#     drawPoints(image, faceLandmarks, 30, 35, True)    # Lower nose
#     drawPoints(image, faceLandmarks, 36, 41, True)    # Left eye
#     drawPoints(image, faceLandmarks, 42, 47, True)    # Right Eye
#     drawPoints(image, faceLandmarks, 48, 59, True)    # Outer lip
#     drawPoints(image, faceLandmarks, 60, 67, True)    # Inner lip

# Check if a point is inside a rectangle
def rect_contains(rectangle, point):
    if point[0] < rectangle[0]:
        return False
    elif point[1] < rectangle[1]:
        return False
    elif point[0] > rectangle[2]:
        return False
    elif point[1] > rectangle[3]:
        return False
    return True


# Draw a point
# def draw_point(img, p, color):
#     cv2.circle(img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0)

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color, landmarks):
    connectivity = np.zeros((len(landmarks), len(landmarks)),dtype=int)
    triangle_list = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangle_list:
        idx1 = landmarks.index((t[0], t[1]))
        idx2 = landmarks.index((t[2], t[3]))
        idx3 = landmarks.index((t[4], t[5]))
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
            connectivity[idx1][idx2] = connectivity[idx1][idx2] + 1
            connectivity[idx2][idx1] = connectivity[idx2][idx1] + 1

            connectivity[idx2][idx3] = connectivity[idx2][idx3] + 1
            connectivity[idx3][idx2] = connectivity[idx3][idx2] + 1
            
            connectivity[idx3][idx1] = connectivity[idx3][idx1] + 1
            connectivity[idx1][idx3] = connectivity[idx1][idx3] + 1
    return connectivity


def delaunay_triangulation(img, landmarks):
    # Define window names
    win_delaunay = "Delaunay Triangulation"

    # Turn off landmark drawing
    draw_landmarks = False

    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)

    # Keep a copy around
    img_orig = img.copy()

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Insert landmarks into subdiv
    for i, l in enumerate(landmarks):
        subdiv.insert(l)

    # Draw delaunay triangles
    connectivity = draw_delaunay(img, subdiv, delaunay_color, landmarks)

    # Draw points
    # if draw_landmarks:
    #     for p in points:
    #         draw_point(img, p, points_color)

    return img, connectivity

# def delaunay_edges(img, landmarks):
#   rect = (0, 0, img.shape[1], img.shape[0])
#   subdiv = cv2.Subdiv2D()
#   subdiv.initDelaunay(rect)
#   for l in landmarks:
#     subdiv.insert(l)
#     edges = subdiv.getEdgeList()
  
#   return edges

def get_edges(file_name):
  img = cv2.imread(file_name)
  landmarks = get_landmarks(img)

  # facePoints(img, landmarks)
  img, connectivity = delaunay_triangulation(img, landmarks)
  # outputNameofImage = "output/image.jpg"
  # cv2.imwrite(outputNameofImage, img)

  num_edge = 0
  edges = []
  ## calculate edge information
  for i in range(connectivity.shape[0]):
    for j in range(connectivity.shape[1]):
      if connectivity[i][j] > 0:
        connectivity[i][j] = 1
        if i <= j:
          num_edge = num_edge + 1
          #########===========註解跟改這兒=======
          # edges.append([i, j])
          edges.append((i, j))
          ########==============================
  return len(edges), edges

########============註解掉這兒===================
# def main():
#   len_edge = []
#   invalid_set = {5, 9, 30, 31, 33, 35, 42, 59, 118, 138, 148, 149, 154, 191}
#   valid_set = []
#   edge_set = {}
#   for i in range(1, 201):
#     if i in invalid_set:
#       continue
#     file_name = f"./frontal_faces/{i}a.jpg"
#     # print(file_name)
#     length, edges = get_edges(file_name)
#     print(edges)
#     len_edge.extend([length])
#     if length == 178:
#       valid_set.extend([i])

#   print(valid_set)
##########=================================


########============每個臉都有的邊============
def main():
  invalid_set = {5, 9, 30, 31, 33, 35, 42, 59, 118, 138, 148, 149, 154, 191}
  valid_set = []
  edge_set = {}
  for i in range(1, 201):
    if i in invalid_set:
      continue
    print(i)
    file_name = f"./frontal_faces/{i}a.jpg"
    # print(file_name)
    length, edges = get_edges(file_name)
    if i==1:
      edge_set = set(edges)
    else:
      edge_set = edge_set.intersection(set(edges))
  
  edge_list = [list(edge) for edge in list(edge_set)]
  print(len(edge_list))
  print(edge_list)
#######====================================
  
  

if __name__ == "__main__":
    main()