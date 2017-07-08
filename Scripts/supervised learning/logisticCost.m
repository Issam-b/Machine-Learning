function [J, grad] = logisticCost(theta, X, y, regOn, lambda)
%LOGISTICCOST Compute cost and gradient for logistic regression with/without regularization
%   J = LOGISTICCOST(theta, X, y, Reg, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.
%   regOn = true to use regularization

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
h = sigmoid(X * theta);

regOn = tolower(regOn);
if Reg == "true"

    theta_reg = [0; theta(2:end)];

    J = 1/m * (-y' * log(h) - (1 - y)' * log(1 - h)) + lambda / (2*m) * (theta_reg' * theta_reg);
    grad = 1/m * (X' * (h - y) + lambda * theta_reg);

else

    J = 1/m * sum( -y .* log(h) - (1 - y) .* log(1 - h) );
    grad = 1/m * (h - y)' * X;
end

end
