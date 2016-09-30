function [theta, mu, sigma, J_history] = linearRegression(X_orig, y, num_iters, alpha, printTheta)
%LINEARREGRESSION Compute linear regression by calculating Theta using either
%   gradient descent or normal equations
%   LINEARREGRESSION(X_orig, y, num_iters, alpha) X_orig should have an intercept element
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
    X = X_orig;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));

    mu = mean(X);
    sigma = std(X);

    for featr = 1:size(X,2) 
        X(:,featr) = ( X(:,featr) - mu(featr) ) / sigma(featr);
    end

    fprintf('Running Gradient Descent ...\n');
    X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
    theta = zeros(size(X,2), 1); % initialize theta

    % compute and display initial cost
    m = length(y); % number of training examples
    J = sum(( X * theta - y ) .^2 )/( 2 * m );


    % run gradient descent
    J_history = zeros(num_iters, 1); % initial J_history as backup of J

    for iter = 1:num_iters

        theta = theta - alpha/m * X' * (X * theta - y);

        % Save the cost J in every iteration
        J_history(iter) = computeCostMulti(X, y, theta);

    end

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
