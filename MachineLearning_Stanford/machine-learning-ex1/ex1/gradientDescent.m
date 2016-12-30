function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

n = length(theta); % the number of thetas
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    temp = (X * theta - y)' * X;
    
    theta = theta - alpha*(X' * (X * theta - y))/m;
    cost = computeCost(X, y, theta);
    disp(sprintf('iter %d, cost is %d',iter,cost));





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = cost;

end

end