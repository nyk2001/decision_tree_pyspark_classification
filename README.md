# decision_tree_pyspark_classification

The notebook uses a decision tree (PySpark) to classify if the breast tumor is malignant or benign. It is a classic example of binary classification. 


Data Set Information
-------------------
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe the characteristics of the cell nuclei present in the image. 

This database is available at : https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from the center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)
