% 5) KNN
knn = fitcknn(X_train, Y_train);
y_knn = zeros(size(true_y));
for x=1:length(cnt)
    y_knn(x) = test_svm(single(cnt(x,:)),S,T,knn);
end

loss = eval_mcr(y_knn(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_knn(indices));
fprintf('The knn mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss,k);
