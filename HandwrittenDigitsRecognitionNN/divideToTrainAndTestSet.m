function [ X_train, X_test, y_train, y_test ] = divideToTrainAndTestSet( X, y )
%divideToTrainAndTestSet split the examples to training set (60%) and test
%set (40%)

    m = size(X,1);

    m_train = round(m * 0.6);
    m_test = size(X,1) - m_train;

    X_train = X(1:m_train, :);
    y_train = y(1:m_train);

    X_test = X(m_train+1:m,:);
    y_test = y(m_train+1:m);


end

