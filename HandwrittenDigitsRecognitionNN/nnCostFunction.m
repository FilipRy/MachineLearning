function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%% Feedforward the neural network and compute J - costs

ones_col = ones(m, 1);
X = [ones_col X];

hiddenLayer = Theta1 * X';

hiddenLayer = sigmoid(hiddenLayer);

ones_row = ones_col';
hiddenLayer = [ones_row; hiddenLayer];

outputLayer = Theta2 * hiddenLayer;
outputLayer = sigmoid(outputLayer);

yMatrix = zeros(size(outputLayer, 1), size(outputLayer, 2));

for i=1:m
    yMatrix(y(i), i) = 1;
end;

costMatrix = -yMatrix .* log(outputLayer) - (1 - yMatrix) .* log(1 - outputLayer);

J = sum(sum(costMatrix));
J = J/m;

regTheta1 = Theta1(:, 2:input_layer_size + 1);
regTheta2 = Theta2(:, 2:hidden_layer_size + 1);

regTerm = sum(sum(regTheta1 .^ 2)) + sum(sum(regTheta2 .^ 2));

J = J + (lambda/(2*m))*regTerm; % J with regularization


%% The backpropagation algorithm to compute the gradients Theta1_grad and Theta2_grad

delta2Matrix = 0;
delta1Matrix = 0;

for t=1:m
    a1 = X(t,:); % 1 * (n + 1)
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);% (hidden_units * 1)
    a2 = [1; a2];% adding bias
    z3 = Theta2 * a2; %10 * 1
    a3 = sigmoid(z3);
    
    delta3 = a3 - yMatrix(:, t);
    delta2 = Theta2' * delta3 .* a2 .* (1 - a2);
    
    delta2 = delta2(2: hidden_layer_size + 1, :);
    
    delta2Matrix = delta2Matrix + delta3 * a2';
    delta1Matrix = delta1Matrix + delta2 * a1;
    
    
end;

% adding regularization
Theta1_grad = (delta1Matrix + lambda * Theta1)/m;

temp1 = delta1Matrix/m;
Theta1_grad(:, 1) = temp1(:, 1);


Theta2_grad = (delta2Matrix + lambda * Theta2)/m;

temp1 = delta2Matrix/m;
Theta2_grad(:, 1) = temp1(:, 1);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
