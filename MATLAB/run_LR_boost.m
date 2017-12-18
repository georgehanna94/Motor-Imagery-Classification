%{
logboost = fitcensemble(X_train, Y_train,'Method', 'LogitBoost');

y_logboost = zeros(size(true_y));
for x=1:length(cnt)
    y_logboost(x) = test_svm(single(cnt(x,:)),S,T,logboost);
ends
%}

tmp = load('y_logboost.mat');
y_logboost = tmp.y_logboost;

loss = eval_mcr(y_logboost(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_logboost(indices));
fprintf('The lr boost mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss,k);
