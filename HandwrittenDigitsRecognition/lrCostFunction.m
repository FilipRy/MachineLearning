function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 


m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


% computing cost function J(theta)
hX = sigmoid(X * theta);

first = -transpose(y) * log(hX);
second = transpose(1-y) * log(1 - hX);

cost = first - second;
J = cost/m;

n = size(theta);


thetaT = theta(2:n,:)';
reg_term = thetaT * theta(2:n,:);
reg_term = reg_term * (lambda/(2*m));

% regularization of J(theta)
J = J + reg_term;


% partial derivates of J(Theta(i)), Theta(i)
gradOne = transpose(hX - y)/m * X;

gradGreater1 = (transpose(hX - y)/m * X) + transpose((lambda/m) * theta);

grad(1) = gradOne(1);
grad(2:n) = gradGreater1(2:n);


grad = grad(:);

end
