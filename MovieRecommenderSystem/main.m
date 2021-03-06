
% PART 1: YOUR RATINGS FOR SOME MOVIES

% loading a movies list from movie_ids.txt
% The dataset movie_ids.txt contains 1682 movies.
movieList = loadMovieList();

% Initializing your ratings of some random movies
your_ratings = zeros(1682, 1);

%Here you can rate some movies you like, based on these ratings this
%recommended system will output other movies you might like.
your_ratings(1) = 4;
your_ratings(98) = 2;
your_ratings(7) = 3;
your_ratings(12)= 5;
your_ratings(54) = 4;
your_ratings(64)= 5;
your_ratings(66)= 3;
your_ratings(69) = 5;
your_ratings(183) = 4;
your_ratings(226) = 5;
your_ratings(355)= 5;

fprintf('\n\n Your ratings:\n');
for i = 1:length(your_ratings)
    if your_ratings(i) > 0 
        fprintf(' You rated %d for %s\n', your_ratings(i), ...
                 movieList{i});
    end
end

% PART 2: TRAIN X AND THETA USING COLLABORATIVE FILTERING

fprintf('\nTraining collaborative filtering...\n');

%  Load data representing the ratings of movies, the ratings are from http://grouplens.org/datasets/movielens/
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [your_ratings Y];
R = [(your_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;

% Training the collaborative filtering model
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

% PART 3: PREDICT RATINGS 

% Using the trained params X and Theta to predict ratings for each movie by
% every user.
p = X * Theta';
my_predictions = p(:,1) + Ymean;

%Giving some movies I might like.
[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(your_ratings)
    if your_ratings(i) > 0 
        fprintf('Rated %d for %s\n', your_ratings(i), ...
                 movieList{i});
    end
end
