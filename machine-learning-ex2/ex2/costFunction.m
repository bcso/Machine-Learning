function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
row = 1;
costSum = 0;

while row <= m
    h = 1./(1 + exp(-1 * theta' * X(row, :)')); % This is our hypothesis (1x1)
    costSum = costSum + (-1 * y(row)*log(h) - (1-y(row))*log(1-h)); % This is the cost of our hypothesis
    
    for j = 1:size(theta)
        % calculate the gradient vector values
        % select each x value as a partial derivative
        grad(j) = grad(j) + (h - y(row)) * X(row, j); 
    end
    
    row = row + 1;
end

J = costSum * (1/m);
grad = grad * (1/m);

return 




% =============================================================

end
