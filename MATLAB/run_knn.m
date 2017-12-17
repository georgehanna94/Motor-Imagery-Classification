% 5) KNN
knn = fitcknn(scaled_train, Y_train);
y_knn = zeros(size(true_y));
for x=1:length(cnt)
    y_knn(x) = test_svm(single(cnt(x,:)),S,T,knn, data_mean, data_std);
end


loss = eval_mcr(y_knn(indices),true_y(indices));
fprintf('The knn mis-classification rate on the test set is %.1f percent.\n',100*loss);
