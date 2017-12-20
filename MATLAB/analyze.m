function [] = analyze(y_true, y_preds, indices, model_str)
relevant_truth = y_true(indices);
relevant_preds = y_preds(indices);
total_true_pos = sum(y_true == 1);
total_true_neg = sum(y_true == -1);
relevant_true_pos = sum(relevant_truth == 1);
relevant_true_neg = sum(relevant_truth == -1);

n = size(y_true, 1);
n_rel = size(relevant_truth, 1);

total_preds_pos = sum(y_preds == 1);
total_preds_neg = sum(y_preds == -1);
relevant_preds_pos = sum(relevant_preds == 1);
relevant_preds_neg = sum(relevant_preds == -1);

fprintf('\n\nAnalyzed values for Truth: Total Pos: %.2f, Total Neg: %.2f, Relevant Pos: %.2f, Relevant Neg: %.2f\n' ...
,(total_true_pos/n), (total_true_neg/n), (relevant_true_pos/n_rel), (relevant_true_neg/n_rel))
fprintf('Analyzed values For %s predictions: Total Pos: %.2f, Total Neg: %.2f, Relevant Pos: %.2f, Relevant Neg: %.2f' ...
    , model_str, (total_preds_pos/n), (total_preds_neg/n), (relevant_preds_pos/n_rel), (relevant_preds_neg/n_rel))

end

