function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); % number of training examples

 
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
J = (1/m)*((-y'*log(h))-(1-y)'*log(1-h)) + (lambda/(2*m))*(theta(2:length(theta))'*theta(2:length(theta)));
mod_theta = theta;
mod_theta(1) = 0;
grad = (1/m)*(h-y)'*X + (lambda/m)*mod_theta';

