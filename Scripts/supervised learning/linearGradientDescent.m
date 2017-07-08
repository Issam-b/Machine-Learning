function [theta, J_history] = linearGradientDescent(X, y, theta, alpha, num_iters)
%LINEARGRADIENTDESCENT Performs gradient descent to learn theta
%   theta = LINEARGRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

theta = zeros(size(X,2), 1); % initialize theta

m = length(y); % number of training examples

% run gradient descent
J_history = zeros(num_iters, 1); % initial J_history as backup of J

for iter = 1:num_iters

    theta = theta - alpha/m * X' * (X * theta - y);

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
