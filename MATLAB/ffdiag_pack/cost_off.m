function [cost]=cost_off(C,V)
%COST_OFF computes diagonalization error as the norm of the off-diagonal elements.
%  C set of matrices in a 3-d array
%  V diagonalizing matrix


[N,N,K]=size(C);

if nargin>1,
  for k=1:K,
    C(:,:,k)=V*C(:,:,k)*V';
  end
end

cost=0;
for k=1:K
  Ck=C(:,:,k);Ck=Ck(:);
  Ck(1:N+1:N*N)=0;
  cost=cost+norm(Ck)^2;
end

return
