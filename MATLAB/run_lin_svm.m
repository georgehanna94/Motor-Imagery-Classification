% Linear SVM
 
lin_svm = fitcsvm(scaled_train, Y_train);
y_svm = zeros(size(true_y));
for x=1:length(cnt)
    y_svm(x) = test_svm(single(cnt(x,:)),S,T,lin_svm, data_mean, data_std);
end

loss = eval_mcr(y_svm(indices),true_y(indices));
fprintf('The lin svm mis-classification rate on the test set is %.1f percent.\n',100*loss);