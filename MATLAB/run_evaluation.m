% Obtains X_train, Y_train, S, T using CSP Feature Extraction
get_features
load data_set_IVb_al_test
load true_labels

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
indices = find(true_y==-1 | true_y==1);
loss = eval_mcr(y_lda(indices),true_y(indices));
fprintf('The LDA mis-classification rate on the test set is %.1f percent.\n',100*loss);

TBE = true_y(indices);
TBP = y_lda(indices);
actual_pos = find(TBE == 1);
actual_neg = find(TBE == -1);
pred_pos_lda = find(TBP == 1);
pred_neg_lda = find(TBP == -1);

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
fprintf('The LR mis-classification rate on the test set is %.1f percent.\n',100*loss);

TBLR = y_lr(indices);
pred_pos_lr= find(TBLR == 1);
pred_neg_lr = find(TBLR == -1);

% 3) SVM

% a) Normalize data
data_mean = mean(X_train);
data_std = std(X_train);
scaled_train = (X_train - data_mean)./data_std;

% Linear SVM

% 5-Fold Cross Validation 
%{

C = logspace(3, 6, 3);
cv_acc = zeros(numel(C),1);
for i=1:numel(C)
    sprintf('Optimizing for C = %f', C(i))
    cv_acc(i) = svmtrain(Y_train, scaled_train, sprintf('-t 0 -c %f -v 5', C(i)))
end

%}
 
lin_svm = fitcsvm(X_train, Y_train);
y_svm = zeros(size(true_y));
for x=1:length(cnt)
    y_svm(x) = test_svm(single(cnt(x,:)),S,T,lin_svm, data_mean, data_std);
end

loss = eval_mcr(y_svm(indices),true_y(indices));
fprintf('The Lin SVM mis-classification rate on the test set is %.1f percent.\n',100*loss);

TBSVM = y_svm(indices);
pred_pos_svm= find(TBSVM == 1);
pred_neg_svm = find(TBSVM == -1);
%{
%% === run pseudo-online ===
oldpos = 1;         % last data cursor
t0 = tic;           % start time
y = []; t = [];     % prediction and true label time series
figure;             % make a new figure
len = 3*nfo.fs;     % length of the display window
speedup = 2;        % speedup over real time
while 1
    % determine data cursor (based on current time)
    pos = 1+round(toc(t0)*nfo.fs*speedup);
    % get the chunk of data since last query
    newchunk = single(cnt(oldpos:pos,:));
    % make a prediction (and also read out the current label)
    y(oldpos:pos) = test_csp(newchunk,S,T,w,b);
    t(oldpos:pos) = true_y(pos);
    % plot the most recent window of data
    if pos > len
        plot(((pos-len):pos)/nfo.fs,[y((pos-len):pos); true_y((pos-len):pos)']);
        line([pos-len,pos]/nfo.fs,[0 0],'Color','black','LineStyle','--');
        axis([(pos-len)/nfo.fs pos/nfo.fs -2 2]);
        xlabel('time (seconds)'); ylabel('class');
        drawnow;
    end
    oldpos = pos;
    
end
%}
