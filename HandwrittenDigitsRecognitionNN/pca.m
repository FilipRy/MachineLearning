function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

U = zeros(n);
S = zeros(n);

% covariance matrix
sigma = X' * X;
sigma = sigma/m;

[U, S, D] = svd(sigma); % computing the singular value decomposition
% U is the size of n * n

end
