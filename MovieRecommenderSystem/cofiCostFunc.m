function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Computing the cost function and gradient for collaborative filtering

hX = X * Theta';
predDiff = hX - Y;

sumExp = (predDiff .* R) .^ 2;

sumExp = sum(sum(sumExp));

regTermTheta = sum(sum(Theta .^ 2)) * (lambda / 2);
regTermX = sum(sum(X .^ 2)) * (lambda / 2);

J = sumExp / 2;% value of cost function J
J = J + regTermTheta + regTermX;% J + regularization

%computing gradient of x
for i=1:num_movies
    usersRated = find(R(i,:) == 1);% Give me all users, who rated movie i
    Theta_temp = Theta(usersRated, :);
    Y_temp = Y(i, usersRated);
    X_grad(i,:) = (X(i,:) * Theta_temp' - Y_temp) * Theta_temp;
    X_grad(i,:) = X_grad(i,:) + lambda * X(i, :);% adding regularization
end;

%computing gradient of theta
for j=1:num_users
    ratedMovies = find(R(:,j) == 1);% Give me all movies rated by user j
    X_temp = X(ratedMovies, :);
    Y_temp = Y(ratedMovies, j);
    Theta_grad(j,:) = (X_temp * Theta(j,:)' - Y_temp)' * X_temp;
    Theta_grad(j,:) = Theta_grad(j,:) + lambda * Theta(j, :);
end;

grad = [X_grad(:); Theta_grad(:)];

end
