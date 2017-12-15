function [f,g] = off(V,C)
%OFF - computes the value and the gradient of the "off" diagonality measure 
% 
% Usage:
%        [f,g] = off(V,C)
%
% code by Pavel Laskov and Andreas Ziehe
% 
% (c) 2004 Fraunfofer FIRST.IDA

F = V*C*V';
f = trace(F'*F) - trace(F.*F);

if nargout > 1
  g = 4*(F-diag(diag(F)))*V*C;
end


