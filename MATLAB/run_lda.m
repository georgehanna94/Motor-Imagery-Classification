% 1) Baseline using LDA
MdlLinear = fitcdiscr(X_train,Y_train);
w = MdlLinear.Coeffs(1, 2).Linear;
b = MdlLinear.Coeffs(1, 2).Const;

y_lda = zeros(size(true_y));
%% == load test data and apply CSP classifier for each epoch ===
for x=1:length(cnt)
    y_lda(x) = test_csp(single(cnt(x,:)),S,T,w,b);
end

% calculate loss
loss = eval_mcr(y_lda(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_lda(indices));
fprintf('The LDA mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss, k);

% For debugging: 
%{
TBE = true_y(indices);
TBP = y_lda(indices);
actual_pos = find(TBE == 1);
actual_neg = find(TBE == -1);
pred_pos_lda = find(TBP == 1);
pred_neg_lda = find(TBP == -1);
%}
