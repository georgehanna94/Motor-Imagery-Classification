% Linear SVM
 
lin_svm = fitcsvm(X_train, Y_train);
y_svm = zeros(size(true_y));
for x=1:length(cnt)
    y_svm(x) = test_svm(single(cnt(x,:)),S,T,lin_svm);
end

loss = eval_mcr(y_svm(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_svm(indices));
fprintf('The linear svm mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss, k);