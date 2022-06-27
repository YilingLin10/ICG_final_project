---
tags: ICG
---
# Final Project
## Environment
* python 3.8
* dlib 19.18.0
```
python -m pip install  https://files.pythonhosted.org/packages/1e/62/aacb236d21fbd08148b1d517d58a9d80ea31bdcd386d26f21f8b23b1eb28/dlib-19.18.0.tar.gz
```
* numpy-1.22.4 
* opencv-python-4.5.5.64
```

pip3 install opencv-python==4.1.2.30  
```
## Dataset
* FEI Face Database
    * https://fei.edu.br/~cet/facedatabase.html
## Feature Extraction
1. Detect facial landmarks
    * dlib: https://www.studytonight.com/post/dlib-68-points-face-landmark-detection-with-opencv-and-python
    ```
    git clone https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
    ```
> ```face_features.py```: get the connectivity of delaunay triangulation of face landmarks
2. Compute distance vectors
    *  Delaunay triangulation
        *  https://github.com/efviodo/dlib-face-recognition-delaunay-triangulation
    *  normalized by the square root of the face area (臉部面積的平方根)
> ```collect_input.py```: collect training data
* 'landmarks'
* 'distances': the distance vector
* 'score'
* 'area_sqrt': the sqare root of the face area
## SVR beauty score model
- [x] 
The SVR defines a smooth function **fb : Rd → R**, which we use to estimate the beauty scores of distance vectors of faces outside the training set.
* Following extensive experimentation, we chose a **Radial Basis Function kernel**, which is capable of modeling the non-linear behavior expected for such a problem. Model selection was performed by a **grid search** over **the width of the kernel σ** , **the slack parameter C** and **the tube width parameter ε**. We used a **soft margin SVR** implemented in **SVMlight** [Joachims 1999].
> ```train_beauty_estimator.py```
## KNN-based beautification
Let ***vi***, and ***bi*** denote the set of distance vectors corresponding to the training set samples, and their associated scores, 
* define the beauty-weighted distances ***wi***, for a given distance vector ***v***, as 
![](https://i.imgur.com/2pRchz3.png)

1. First sorting {***vi***} such that ***wi ≥ wi+1*** (descending order)
2.  searching for the value of K maximizing the SVR beauty score ***fb*** of the weighted sum
![](https://i.imgur.com/cr5WRd7.png)
3. linearly interpolating between v and v' instead of replacing v with v' (先沒有做)
    * https://docs.scipy.org/doc/scipy/tutorial/interpolate.html
> ```knn_beautification.py```
## Distance Embedding and Warping
* Objective:  Convert v' to a set of new facial landmarks.
* Obtained by minimizing E, where
![](https://i.imgur.com/BhpohcS.png)
    * ei,j is our facial mesh connectivity matrix
    * αi,j = 1
        * for edges that connect feature points from different facial features
    * αi,j = 10
        * for edges connecting points belonging to the same feature. 
    * di,j is the entry in v' corresponding to the edge ei,j.
* Use the Levenberg-Marquardt (LM) algorithm 
    * ```scipy.optimize.least_squares```

* ```method{‘trf’, ‘dogbox’, ‘lm’}``` , optional
            * Algorithm to perform minimization. 
            * ‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK. Doesn’t handle bounds and sparse Jacobians. Usually the most efficient method for small unconstrained problems.
            * https://pythonmana.com/2020/12/20201210164251696e.html


### Warping
* tfa.sparse_image_warp

### Results：
* 13
![](https://i.imgur.com/2ECLyjX.jpg)
* 20
* 25
![](https://i.imgur.com/9FALt46.jpg)
* 27
![](https://i.imgur.com/XIQoKKw.jpg)
* 32
![](https://i.imgur.com/eY5hSOF.jpg)
* 37
![](https://i.imgur.com/ToYkcm9.jpg)
* 47
![](https://i.imgur.com/caJQG7w.jpg)
* 52
![](https://i.imgur.com/6YTq9Fu.jpg)

