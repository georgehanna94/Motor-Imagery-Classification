function W = getW(C)
%getW   computes the update for FFDIAG.
% 
% Usage:
%    function W = getW(C)
%
% Input: 
%    C -  3-d array of matrices C_k
%
%  Output:
%    W - update matrix with zeros on the main diagonal
%
% See also:
%          FFDIAG
%

  
[m,n,K] = size(C);
W = zeros(n);
z = zeros(n);
y = zeros(n);

for i=1:n
  for j=1:n
    
    for k=1:K
      z(i,j) = z(i,j)+C(i,i,k)*C(j,j,k);
      y(i,j) = y(i,j)+0.5*C(j,j,k)*(C(i,j,k)+conj(C(j,i,k)));
    end
  end
end

for i=1:n-1
  for j=i+1:n
    W(i,j) = (z(j,i)*y(j,i)-z(i,i)*y(i,j))/(z(j,j)*z(i,i)-z(i,j)^2);
    W(j,i) = ((z(i,j)*y(i,j)-z(j,j)*y(j,i))/(z(j,j)*z(i,i)-z(i,j)^2));
  end
end
    
