% 4) RBF Kernel 
 
RBFSVM = fitcsvm(X_train,Y_train,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
y_rbf = zeros(size(true_y));
for x=1:length(cnt)
    y_rbf(x) = test_svm(single(cnt(x,:)),S,T,RBFSVM);
end

loss = eval_mcr(y_rbf(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_rbf(indices));
fprintf('The rbf svm mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss, k);

% For Testing Purposes 
%{

TBRBF = y_rbf(indices);
pred_pos_rbf= find(TBRBF == 1);
pred_neg_rbf = find(TBRBF == -1);
%}