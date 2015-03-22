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

costSum = 0;
thetaSum = 0;

% =============== Calculate the cost function =================
for j = 2:size(theta)
    thetaSum = thetaSum + theta(j) ^ 2;
end    

for row = 1:m
    h = 1./(1 + exp(-1 * theta' * X(row, :)')); % This is our hypothesis (1x1)
    costSum = costSum + (-1 * y(row)*log(h) - (1-y(row))*log(1-h)); % This is the cost of our hypothesis
end

J = costSum * (1/m) + (lambda*(1/(2*m))) * thetaSum;

% ================ Calculate the gradient vector ===============

% For theta_j where j = 0
for row = 1:m
    h = 1./(1 + exp(-1 * theta' * X(row, :)')); % This is our hypothesis (1x1)
    grad(1) = grad(1) + (h - y(row)) * X(row, 1);
end
grad(1) = (1/m) * grad(1);

% For theta_j where j >= 1
for j = 2:size(theta)
    for row = 1:m
        h = 1./(1 + exp(-1 * theta' * X(row, :)')); % This is our hypothesis (1x1)
        grad(j) = (grad(j) + (h - y(row)) * X(row, j)); 
    end
    grad(j) = (1/m) * (grad(j) + (lambda) * theta(j));
end

return

 






% =============================================================

end
