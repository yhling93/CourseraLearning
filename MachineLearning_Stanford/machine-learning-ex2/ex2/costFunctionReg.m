function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


part1 = -y.*log(sigmoid(X*theta));
part2 = (1-y).*log(1-sigmoid(X*theta));
partialTheta = theta(2:length(theta)).*theta(2:length(theta));
J = sum(part1-part2)/m + sum(partialTheta)*lambda/(2*m);


grad = (X'*(sigmoid(X*theta)-y))/m + lambda*theta/m;
grad(1) = X(:,1)'*(sigmoid(X*theta)-y)/m;


% =============================================================

end
