function y = test_csp(X,S,T,w,b)
% Apply CSP classifier to a new chunk of data
% Prediction = test_csp(Raw-Block, Spatial-Flt, Temporal-Flt, Weights, Bias)
%
% In:
%   X : chunk of data [#samples x #channels]
%
%   S : spatial filter as computed by train_csp [#channels x #filters]
%
%   T : temporal filter as computed by train_csp (#samples x #samples)
%
%   w : linear classifier weights
%
%   b : linear classifier bias

% append new chunk X to a buffer B (B has same #samples as the temporal filter)
global B;
if any(size(B) ~= [length(T),length(S)])
    B = zeros(length(T),length(S));
end
B = [B;X]; B = B(end-length(T)+1:end,:);

% apply temporal and spatial filters to the buffer B, calculate log-variance, 
% and apply the linear classifier (TODO: fill in...)
X = log(var(T*(B*S')));
%filt_data = T*X; %Supposed to filter data from high freq content

y = sign(X*w+b);
