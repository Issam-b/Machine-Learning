function [Theta1 Theta2] = neuralNet(X, y, num_iters, lambda, input_layer_size, hidden_layer_size, output_layer_size, printTheta)
%NEURALNET return the parameters Theta1 and Theta2 computed using neural networks that has 3 layers
%   NEURALNET(X, y, num_iters, lambda, input_layer_size, hidden_layer_size, output_layer_size, printTheta) X should have an intercept term
%   num_iters = number of iterations for gradient descent only (ex. 400)
%   lambda = regularization term (ex. 0.5, you should also try other values for better results)
%   input_layer_size = number of units in input layer
%   hidden_layer_size = number of units in hidden layer
%   output_layer_size = number of units in output layer
%   printTheta = true to print Theta to screen
%
%   By: Issam


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Train the neural network
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', num_iters);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCost(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

% Printing theta
printTheta = tolower(printTheta);
if printTheta == "true"
    fprintf('Theta:\n');
    disp(Theta1);
    disp(Theta2);
end
