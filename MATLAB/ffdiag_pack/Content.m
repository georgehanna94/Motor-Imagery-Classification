% This MATLAB code implements the FFDIAG algorithm for joint approximate diagonalization of several matrices.
% The algorithm is based on the Frobenius-norm formulation of the joint diagonalization problem, and
% addresses diagonalization with a general, non-orthogonal
% transformation. The iterative scheme of the algorithm is based on a
% multiplicative update which ensures the invertibility of the
% diagonalizer.
%  
% The details on the derivation of the algorithm as well as comparisons
% to other algorithms for joint diagonalization are presented
% in the paper:   
%  A. Ziehe, P. Laskov, G. Nolte and K.-R. Mueller, 
% "A Fast Algorithm for Joint Diagonalization with Non-orthogonal
%  Transformations and its Application to Blind Source Separation"
%  Journal of Machine Learning Research vol 5, pages 777-800, 2004.
%
% An earlier version has been published under the name LSDIAG 
% at the ICA 2003 Workshop in Nara, Japan:  
% 
%  A. Ziehe and P. Laskov and K.-R. Mueller and G. Nolte
%  "A Linear Least-Squares Algorithm for Joint Diagonalization",
%  in Proc. of the 4th International Symposium on 
%  Independent Component Analysis and Blind Signal Separation
%  (ICA2003), pages 469--474, Nara, Japan 2003.
% 
%  
% See also
% http://www.first.fraunhofer.de/~ziehe/research/FFDiag.html
%
% To get started type "help ffdiag" in the MATLAB command window.
%

