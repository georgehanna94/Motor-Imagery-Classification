function [f,g] = get_off(V,C)
%GET_OFF  a wrapper for OFF 
%
% This function loops over 'off' and  adds up things over multiple matrices C.

[m,n,K] = size(C);
f = 0;
g = sparse(n,n);
h = sparse(n^2,n^2);
h1 = sparse(n^2,n^2);
h2 = sparse(n^2,n^2);

for k=1:K
  if nargout <= 1
    f = f + off(V,C(:,:,k));
  elseif nargout == 2
    [ff,gg] = off(V,C(:,:,k));
    f = f+ff;
    g = g+gg;
    gg
  end
end
