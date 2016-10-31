function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

centroids = zeros(K, n);

centr_usage_count = zeros(K);

for i=1:m
    u_c = idx(i);
    centroids(u_c,:) = centroids(u_c,:) + X(i,:);
    centr_usage_count(u_c) = centr_usage_count(u_c) + 1;
end;

for i=1:K
    centroids(i, :) = centroids(i, :) ./ centr_usage_count(i);
end;





% =============================================================


end

