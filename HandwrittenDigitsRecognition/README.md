## HandwrittenDigitsRecognition using oneVsAll and logistic regression

6000 examples[1] have been used to train the classifier. Each example represent a preprocessed handwritten digit represented as an gray-scale image, 28 pixels in height and 28 pixels in width. 784 pixels in total are representing 784 features, which is a quite large number. Therefore I used PCA to reduce the number of features. I found out that ca 100 features are enough to retain more than 95% of the covariance matrix (see reduceFeaturesPCA.m).
After applying PCA there was a small difference in the train, test set accurancy.  
Train set: with PCA I received 82.85 % accuracy, without PCA 85.383333%.  
Test set: with PCA I received 80.95 % accuracy, without PCA 80.55%.

[1] https://www.kaggle.com/c/digit-recognizer/data
