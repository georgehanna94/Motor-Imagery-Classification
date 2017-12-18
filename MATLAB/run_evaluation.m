% Obtains X_train, Y_train, S, T using CSP Feature Extraction
get_features
load data_set_IVb_al_test
load true_labels
indices = find(true_y==-1 | true_y==1);

run_dec_tree
run_lda
run_LR
run_lin_svm
run_rbf_svm
run_knn
run_ann 

analyze(true_y, y_dec_tree, indices, 'Decision Tree')
analyze(true_y, y_lda, indices, 'LDA')
analyze(true_y, y_lr, indices, 'LR')
analyze(true_y, y_svm, indices, 'Lin SVM')
analyze(true_y, y_rbf, indices, 'RBF SVM')
analyze(true_y, y_knn, indices, 'KNN')
analyze(true_y, y_ann, indices, 'ANN')

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
