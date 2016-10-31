## Image compression using K-means algorithm

There are two main steps done in the K-means algorithm
- Assignment step: each example in the data set is assigned to the nearest centroid
- Move step: the location of each centroid is computed as an average of the examples assigned to this centroid

For the image compression we have $image_width * $image_height examples in the training set. Each example represents a pixel of the image. 16 of them are randomly chosen as the centroids. Each pixel is then assigned to one centroid. All pixels within one centroid have the same color, which means that the output picture has only 16 colors. So only 4bits are used to represent each pixel.

This exercise was done as one of the programming assignments in machine learning course at coursera [1].


[1] https://www.coursera.org/learn/machine-learning