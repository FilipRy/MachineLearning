function [ X_train, X_test ] = reduceFeaturesPCA( X_train, X_test )
%reduceFeaturesPCA Uses the PCA to reduce the # of features.

    n = size(X_train,2);
    [U, S] = pca(X_train); %computing pca on the training set
    
    %% Finding the optimal k
    vector_S = zeros(n);
    
    for i=1:n
        vector_S(i) = S(i,i);
    end;
    
    best_k = n;
    sum_to_k = sum(vector_S);
    
    for k=n-1:-1:1
        retained_variance = sum_to_k/sum(vector_S);
        if retained_variance < 0.95 % want to retain at least 95% of the variance
            best_k = k + 1;
            break;
        end;
        sum_to_k = sum_to_k - vector_S(k);
    end;
    
    k = best_k;
    
    %% Reducing the number of features using the found k
    U_reduce = U(:,1:k); 
    
    X_train = U_reduce' * X_train'; %reducing the number of features in the training set
    
    X_test = U_reduce' * X_test'; % reducing the number of features in the test set, by using the same U_reduce as for X_train was used
    
    % X_train and X_test are size of k * m, but we want them m * k
    X_train = X_train';
    X_test = X_test';


end

