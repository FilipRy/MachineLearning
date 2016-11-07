function [ X,y ] = loadExamples( filename )
%loadExamples load examples from a CSV file

    trainData = csvread(filename);

    % training examples are loaded from a .csv containing 1+28*28 columns (first column is a label) each
    % representing the lightness or darkness of that pixel, with higher numbers meaning darker. 
    % This pixel-value is an integer between 0 and 255, inclusive.
    
    num_features = size(trainData, 2) - 1; % -1 because the very first column are the labels;

    X = trainData(:,2:num_features+1);

    y = trainData(:,1);

    for i=1:size(y,1) % mapping the digit 0 to label 10
        if y(i)==0
            y(i)=10;
        end;
    end;

end

