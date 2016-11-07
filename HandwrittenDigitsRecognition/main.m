
%% Initialization
clear ; close all; clc

%% load training examples

num_labels = 10;% we have digits 0-9

%the examples.csv are obtained from https://www.kaggle.com/c/digit-recognizer/data
[X, y] = loadExamples('examples.csv');

X = X / 255;

% m = around 10000
% n = 28 * 28, which is quite large, so we apply PCA to reduce the number
% of features


[ X_train, X_test, y_train, y_test ] = divideToTrainAndTestSet( X, y );


%% applying PCA 
[X_train, X_test] = reduceFeaturesPCA(X_train, X_test);

%% run multiclass classification by utilizing logistic regression
lambda = 1;

% fmincg is used to minimize the cost function J(theta)
[all_theta] = oneVsAll(X_train, y_train, num_labels, lambda);

%% predictions

predictedDigitsTrain = predictOneVsAll(all_theta, X_train);


% with PCA we received 82.85 % accuracy, without PCA 85.383333%
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predictedDigitsTrain == y_train)) * 100);

predictedDigitsTest = predictOneVsAll(all_theta, X_test);

% with PCA we received 80.95 % accuracy, without PCA 80.55%
fprintf('\nTest Set Accuracy: %f\n', mean(double(predictedDigitsTest == y_test)) * 100);

