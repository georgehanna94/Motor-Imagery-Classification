% Evaluation takes a while, the prediction mat has been saved to workspace
% and loaded to save time:
tmp = load('y_ann.mat');
y_ann = tmp.y_ann;
%{
net = patternnet(10);
YANN = [(Y_train == -1), (Y_train == 1)];
net = train(net,transpose(X_train),transpose(YANN));

y_ann = zeros(size(true_y));
for x=1:length(cnt)
    fprintf('\nComputation %f/%f', x, length(cnt))
    y_ann(x) = test_ann(single(cnt(x,:)),S,T,net);
end
%}

loss = eval_mcr(y_ann(indices),true_y(indices));
k = compute_cohens_k(true_y(indices), y_ann(indices));
fprintf('The Neural Network mis-classification rate on the test set is %.2f percent & kappa = %.3f\n',100*loss, k);