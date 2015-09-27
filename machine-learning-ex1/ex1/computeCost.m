function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
%regression = X*theta;
%error = regression - y;
%J = (1/(2*length(y)) * dot(error,error) );

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% theta => vector [2x1]
% X => matrix [97x2]
% H => vector [97x1]

% H = ( theta' * X' )' => hypothetis
% J = 1/2m * sum( h(theta) - y )^2 => our cost function

H = ( theta' * X' )';
J = 1/( 2 * m ) *  sum( ( H - y ) .^ 2 );


% =========================================================================

end
