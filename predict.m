function p = predict(theta, X)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);


h_theta = sigmoid(X*theta);
ln = length(h_theta);
for i=1:ln
    if(h_theta(i)>=0.5)
        p(i) = 1;
    else
        p(i) = 0;
    end
end

end
