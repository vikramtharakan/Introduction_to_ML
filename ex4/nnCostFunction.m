function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Reshaping y
newy = zeros(numel(y), max(y));
newy(sub2ind(size(newy), 1:numel(y), y')) = 1;

% You need to return the following variables correctly 
X = [ones(m,1), X];
a2 = sigmoid(X*Theta1');
a2 = [ones(m,1), a2];
h_x2 = sigmoid(a2*Theta2');

J = 0;

for i=1:size(newy,2)
    J = J -1/m*(newy(:,i)'*log(h_x2(:,i)) + (1-newy(:,i)')*log(1-h_x2(:,i)));
end
% Regularization
J = J + lambda/(2*m)*( sum(sum(Theta1(:,(2:end)).^2)) + sum(sum(Theta2(:,(2:end)).^2)) );

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


grad_1 = 0;
grad_2 = 0;


for t = 1:m
    %ForwardProp
    a1 = X(t,:)';
    yt = newy(t,:)';
    a2 = sigmoid(Theta1*a1);
    a2 = [1; a2];
    a3 = sigmoid(Theta2*a2);

    yy = ([1:num_labels]==y(t))';
    S3 = a3 - yy;
    S2 = Theta2'*S3.*[1; sigmoidGradient(Theta1*a1)];
    S2 = S2(2:end); 
    
    Theta1_grad = Theta1_grad + S2 * a1';
	Theta2_grad = Theta2_grad + S3 * a2';

end

Theta1_grad = (1/m)*Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m)*Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];


%size(grad_1)        %-- 5x1 or 4x1
%size(grad_2)        %-- 4x1
%disp(size(Theta1)) %-- 4x3
%disp(size(Theta2)) %-- 4x5


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
