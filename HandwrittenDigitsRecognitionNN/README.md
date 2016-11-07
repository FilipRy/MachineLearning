## HandwrittenDigitsRecognition using neural network

6000 examples [1] have been used to train the classifier. Each example represent a preprocessed handwritten digit represented as an gray-scale image, 28 pixels in height and 28 pixels in width. 784 pixels in total are representing 784 features, which is a quite large number. Therefore I used PCA to reduce the number of features. I found out that ca 100 features are enough to retain more than 95% of the covariance matrix (see reduceFeaturesPCA.m).
After applying PCA there was a small difference in the train, test set accurancy.  

with PCA:
- Training Set Accuracy: 95.300000
- Test Set Accuracy: 90.000000

without PCA:
- Training Set Accuracy: 99.900000
- Test Set Accuracy: 93.150000


[1] https://www.kaggle.com/c/digit-recognizer/data