## Recommender Systems for Movie Ratings

I implemented collaborative filtering algorithm to predict ratings for given movies. The dataset [1] used for this exercise contains 1682 movies, each rated by at most 943 users.   
The collaborative filtering algorithm minimizes cost function by utilizing gradient descent or similar algorithm (fmincg algorithm was used for this exercise). The cost function is simply said a difference between a predicted rating of movie m by a user u and the actual rating of m by u.  
After training we have params:
- X as feature vector for movies
- Theta as parameter vector for users

By utilizing Theta and X params we recommend a set of movies a particular user may like.

I have implemented this exercise as one of the programming assignments in machine learning course at coursera [2].  


[1] http://grouplens.org/datasets/movielens/  
[2] https://www.coursera.org/learn/machine-learning
