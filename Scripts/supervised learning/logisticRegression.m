function [theta, J] = logisticRegression(X, y, num_iters, regOn, lambda, num_labels, degree, printTheta)
%LOGISTICREGRESSION Compute logistic regression by calculating Theta
%   LOGISTICREGRESSION(X, y, num_iters, regOn, lambda, num_labels, degree, printTheta) X should have an intercept term
%   num_iters = number of iterations for gradient descent only (ex. 1500)
%   regOn = true if using regularization
%   lambda = regularization term
%   num_labels = number of labels to train (or classes)
%   degree = number of features to add using feature mapping
%   printTheta = true to print Theta to screen
%
%   By: Issam

%%TODO: expand the plot Function
% plot data (for 2D only for now)
logisticPlot(X, y);

[m, n] = size(X); % useful variables

% Choose between 2 labels classifier or multiple class classifier
if num_labels < 3

    %%TODO: expand map feature for more than 2 labels
    % use map feature to add more features to the input
    out = ones(size(X1(:,1)));
    for i = 1:degree
        for j = 0:i
            out(:, end+1) = (X1.^(i-j)).*(X2.^j);
        end
    end

    theta = zeros(n + 1, 1);

    % Optimize using fminunc
    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter', num_iters);
    [theta, J, exit_flag] = ...
            fminunc(@(t)(logisticCost(t, X, y, regOn, lambda)), theta, options);

else if num_labels >= 3

    theta = zeros(num_labels, n + 1);
    J = zeros(num_labels, 1);

    for i = 1:num_labels

        % Set Initial theta
        theta_init = zeros(n + 1, 1);

        % Set options for fminunc
        options = optimset('GradObj', 'on', 'MaxIter', num_iters);

        % Run fmincg to obtain the optimal theta
        [theta_temp J_temp] = ...
            fmincg (@(t)(logisticCost(t, X, (y == i), regOn, lambda)), ...
                    theta_init, options);

        theta(i,:) = theta_temp;
        J(i) = J_temp;
    end

else

    fprintf('Incorrect number of labels\n');

end

% Printing theta
printTheta = tolower(printTheta);
if printTheta == "true"
    fprintf('Theta:\n');
    disp(theta);
    fprintf('Cost:\n');
    disp(J);
end
