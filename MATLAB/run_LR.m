% 2) Logistic Regression
Y_LR = (Y_train == 1) + 1;
LR = mnrfit(double(X_train), Y_LR);

y_lr = zeros(size(true_y));
for x=1:length(cnt)
    y_lr(x) = test_csp(single(cnt(x,:)),S,T,LR(2:end),LR(1));
end

% calculate loss
indices = find(true_y==-1 | true_y==1);
loss = eval_mcr(y_lr(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_lr(indices));
fprintf('The LR mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss, k);

% For debugging
%{
TBLR = y_lr(indices);
pred_pos_lr= find(TBLR == 1);
pred_neg_lr = find(TBLR == -1);
%}