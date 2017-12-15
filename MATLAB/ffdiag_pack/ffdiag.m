function [V,C,stat] = ffdiag(C0,V0)
%FFDIAG  Diagonalizes a set of matrices C_k with a (single) non-orthogonal transformation. 
%
%
% Usage:
%
%  function [V,CD,stat] = ffdiag(C0,V0)  
%
% Input:
%    C0 - N x N x K array containing K symmetric matrices C_k which are to be diagonalized.
%    V0 - initial value for the diagonalizer (default eye(N))
%
% Output:
%    V    - joint diagonalization transformation (dimension N x N)
%    CD   - diagonalized set of matrices (dimension N x N x K).
%    stat - structure with some statistics of the algorithm's performance
%        .etime      - elapsed time
%        .errdiag    - diagonalization error
%  
% code by Andreas Ziehe and Pavel Laskov
%
%  (c) 2004  Fraunhofer FIRST.IDA 
%
% Usage example:  
%  
% gen_mat; [V,C,stat]=ffdiag(C0); imagesc(V*A);


  
%  
% The algorithm is derived in the paper:   
%  A. Ziehe, P. Laskov, G. Nolte and K.-R. Mueller, 
% "A Fast Algorithm for Joint Diagonalization with Non-orthogonal
%  Transformations and its Application to Blind Source Separation"
%  Journal of Machine Learning Research vol 5, pages 777-800, 2004.
%
% An earlier version has been presented at the ICA 2003 Workshop in
% Nara, Japan  
% 
%  A. Ziehe and P. Laskov and K.-R. Mueller and G. Nolte
%  "A Linear Least-Squares Algorithm for Joint Diagonalization",
%  in Proc. of the 4th International Symposium on 
%  Independent Component Analysis and Blind Signal Separation
%  (ICA2003), pages 469--474, Nara, Japan 2003.
% 
%  
% See also
% http://www.first.fraunhofer.de/~ziehe/research/ffdiag.html

  
  eps = 1e-9;

%theta = 0.1;    % threshold for stepsize heuristics 

[m,n,k] = size(C0);
C = C0;
Id=eye(n);
V = Id;

%more clever initialization ???
%[V,D]=eig(C0(:,:,1),C0(:,:,2));
%or
%[V,D]=eig(sum(C0,3));
%V=V';

inum = 1;
df = 1;

stat.method='FFDIAG';
tic;
while (df > eps & inum < 500)
  
  % Compute W
  W = getW(C);

  %old normalization  no longer recommended 
  %as it deteriorates  convergence
  % 
  %if (norm(W,'fro') > theta)
  %  W = theta/norm(W,'fro')*W;
  %end

  % A much better way is to 
  % scale W by power of 2 so that its norm is <1 .
  % necessary to make approximation assumptions hold
  % i.e. W should be small
  % 
  
  %if 0,
  [f,e] = log2(norm(W,'inf'));
  % s = max(0,e/2);

  s = max(0,e-1);
  W = W/(2^s );
  
    
  
  
  % Compute update
  V = (Id+W)*V;

  %re-normalization
  V=diag(1./sqrt(diag(V*V')))*V;  %norm(V)=1 
  
  for i=1:k
    C(:,:,i) = V*C0(:,:,i)*V';
  end
   
  % Save stats
  stat.W(:,:,inum) = W;
  [f] = get_off(V,C0);

  stat.f(inum) = f;
  stat.errdiag(inum)=cost_off(C0,normit(V')'); 
  stat.nw(inum) = norm(W(:));
  
  
  if inum > 2
    df = abs(stat.f(inum-1)-stat.f(inum));

  end
  
  % fprintf(1,'itr %d :  %6.8f  \n ',inum, f);
  inum = inum+1;
end

stat.etime=toc;   %elapsed time
stat.niter=inum;  %number of iterations
stat.V=V;         %estimated diagonalizer

%subplot(2,1,1);
%semilogy(stat.f);
%title('objective function');

%subplot(2,1,2);
%semilogy(stat.nw');
%title('norm W');

