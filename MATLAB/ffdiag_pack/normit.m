function V=normit(V)
%NORMIT  Normalize the Frobenius norm of all columns of a matrix to 1. 
[n,m]=size(V);

for k=1:n,
  nn=norm(V(:,k),'fro');
  if nn<eps,warning('divsion may cause numerical errors');end
  V(:,k)=V(:,k)/nn;
end

