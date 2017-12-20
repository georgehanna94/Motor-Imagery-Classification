tree = fitctree(X_train, Y_train);

y_dec_tree = zeros(size(true_y));
for x=1:length(cnt)
    y_dec_tree(x) = test_svm(single(cnt(x,:)),S,T,tree);
end

loss = eval_mcr(y_dec_tree(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_dec_tree(indices));
fprintf('\nThe decision tree mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss, k);