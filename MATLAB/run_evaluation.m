% Obtains X_train, Y_train, S, T using CSP Feature Extraction
get_features

% 1) Baseline using LDA
MdlLinear = fitcdiscr(X_train,Y_train);
w = MdlLinear.Coeffs(1, 2).Linear;
b = MdlLinear.Coeffs(1, 2).Const;

%% == load test data and apply CSP classifier for each epoch ===
load data_set_IVb_al_test
for x=1:length(cnt)
    y(x) = test_csp(single(cnt(x,:)),S,T,w,b);
end

% calculate loss
load true_labels
indices = true_y==-1 | true_y==1;
loss = eval_mcr(sign(y(indices)),true_y(indices)');
fprintf('The mis-classification rate on the test set is %.1f percent.\n',100*loss);

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
