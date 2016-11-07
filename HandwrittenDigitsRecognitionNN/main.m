
%% Initialization
clear ; close all; clc

%%  load training examples

[X, y] = loadExamples('examples.csv');

input_layer_size = size(X, 2); % each pixel of an image
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;% we have digits 0-9

% n = input_layer_size = 28 * 28, which is quite large, so we apply PCA to
% reduce the number of features

X = X / 255;

[ X_train, X_test, y_train, y_test ] = divideToTrainAndTestSet( X, y );

%% applying PCA 
%[X_train, X_test] = reduceFeaturesPCA(X_train, X_test);

%input_layer_size = size(X_train, 2); % we must change input_layer_size after PCA
%hidden_layer_size = round(input_layer_size /10);

%%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% Training NN
%  To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc".
fprintf('\nTraining Neural Network... \n')


options = optimset('MaxIter', 400);


lambda = 1;

% reference to costFunction
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

                               
% minimizing costfunction J(theta)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
             
             

predictedDigitsTrain = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(predictedDigitsTrain == y_train)) * 100);

predictedDigitsTest = predict(Theta1, Theta2, X_test);

fprintf('\nTest Set Accuracy: %f\n', mean(double(predictedDigitsTest == y_test)) * 100);

% with PCA
% Training Set Accuracy: 95.300000
% Test Set Accuracy: 90.000000

             