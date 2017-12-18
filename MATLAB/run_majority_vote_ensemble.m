dec_tree_preds = y_dec_tree(indices);
ann_preds = y_ann(indices);
lr_preds = y_lr(indices);
rbf_preds = y_rbf(indices);
svm_preds = y_svm(indices);
all_preds = [rbf_preds, svm_preds, ann_preds,dec_tree_preds, lr_preds];
y_maj_vote = mode(all_preds, 2);

% calculate loss
loss = eval_mcr(y_maj_vote,true_y(indices));
k = compute_cohens_k(true_y(indices), y_maj_vote);
fprintf('The Majority Vote ensemble mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss, k);