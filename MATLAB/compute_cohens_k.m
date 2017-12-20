function [ k ] = compute_cohens_k(true_y, predicted_y)
% Assuming binary classification & class labels are -1 & 1:
n = size(true_y, 1);
p_o = sum(true_y == predicted_y)/n;

pos_term = (sum(true_y == 1)*sum(predicted_y == 1))/n;
neg_term = (sum(true_y == -1)*sum(predicted_y == -1))/n;
p_e = (pos_term + neg_term)/n;

k = (p_o - p_e)/(1 - p_e);
end

