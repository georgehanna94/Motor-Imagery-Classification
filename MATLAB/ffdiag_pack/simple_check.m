%simple_check   runs a very simple test of FFDIAG on two matrices and
%               compares the result with MATLAB's eigenvalue solver
%               Please note that this is only a sanity check. FFDIAG
%               is NOT supposed to replace the eigenvalue solver.

clear;

disp('Simple check of FFDIAG')

disp('Loading test data')

load   test_problem_K2.mat


%solve the joint diagonalization problem 
%with EIG (general eigenvalue problem)
disp('Solving general eigenvalue problem')
[VV,DD]=eig(C0(:,:,1),C0(:,:,2));
 



% solve the same problem with FFDIAG
disp('Running FFDIAG ...')
disp('iteration       off')
[V,C,stat]=ffdiag(C0);

disp('the error should have converged to zero.')

% compare the results
disp('... and the following matrix should be close to a permutation matrix')
 abs(normit(V*inv(VV')))

 figure(1)
 semilogy(stat.errdiag) 
 xlabel('iterations')
 ylabel('off')
 disp('done.')