% Takes along time, predictions been saved 
%{ 
random_forest = fitcensemble(X_train, Y_train,'Method', 'Bag');

y_random_forest = zeros(size(true_y));
for x=1:length(cnt)
    y_random_forest(x) = test_svm(single(cnt(x,:)),S,T,random_forest);
end
%}
tmp = load('y_random_forest.mat');
y_random_forest = tmp.y_random_forest;
loss = eval_mcr(y_random_forest(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_random_forest(indices));
fprintf('The Random Forest mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss, k);