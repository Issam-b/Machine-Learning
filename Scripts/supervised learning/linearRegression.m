function [theta, mu, sigma, J_history] = linearRegression(X_orig, y, num_iters, alpha, printTheta)
%LINEARREGRESSION Compute linear regression by calculating Theta using either
%   gradient descent or normal equations
%   LINEARREGRESSION(X_orig, y, num_iters, alpha) return theta, mean mu and standard deviation sigma, and vector of cost values
%   num_iters = number of iterations for gradient descent only (ex. 1500)
%   alpha = the learning rate (ex. 0.01)
%   printTheta = true to print Theta to screen
%
%   By: Issam


% Choose gradient descent or normal equations
fprintf("Use Gradient descent or Normal equations (Y / N): ");
grad_norm = yes_or_no();

if grad_norm == 1

    % Feature normalization
    [X, mu, sigma] = featureNormalize(X_orig);
    X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

    fprintf('Running Gradient Descent ...\n');
    [theta, J_history] = linearGradientDescent(X, y, theta, alpha, num_iters);

else
    % setting gradient descent parameters to 0
    mu = 0;
    sigma = 0;
    J_history = 0;

    fprintf('Running Normal equations method ...\n');
    theta = zeros(size(X, 2), 1);
    theta = pinv(X' * X) * X' * y;
end

printTheta = tolower(printTheta);
if printTheta == "true"
    % print theta to screen
    fprintf('Theta found:\n');
    disp(theta);
end
