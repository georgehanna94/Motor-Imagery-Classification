function [X,Y,S,T] = extract_csp_features(EEG,Fs,mrk,wnd,f,nof,n)
fprintf('Extracting features using CSP')

% do frequency filtering using FFT
[t,c] = size(EEG); idx = reshape(1:t*c-mod(t*c,n),n,[]);
FLT = real(ifft(fft(EEG).*repmat(f(Fs*(0:t-1)/t)',1,c)));

% estimate temporal filter using least-squares
T = FLT(idx)/EEG(idx);

% extract data for all epochs of the first class concatenated (EPO{1}) and 
% all epochs of the second class concatenated (EPO{2})
% each array is [#samples x #channels]
wnd = round(Fs*wnd(1)) : round(Fs*wnd(2));

for k = 1:2
    EPO{k} = FLT(repmat(find(mrk==k),length(wnd),1) + repmat(wnd',1,nnz(mrk==k)),:);
end

% calculate the spatial filter matrix S using CSP (TODO: fill in)
C_1 = cov(EPO{1});
C_2 = cov(EPO{2});

%Form 3D cov matrix
R = zeros(2,118,118);
R(1,:,:)=C_1;
R(2,:,:)=C_2;

%Get projection matrix
S = MulticlassCSP(R,2*nof);

% log-variance feature extraction
for k = 1:2
    X{k} = squeeze(log(var(reshape(EPO{k}*S', length(wnd),[],2*nof))));
end

class_1_target = ones(length(X{1}),1);
class_2_target = -1.*ones(length(X{2}),1);
Y = vertcat(class_1_target,class_2_target);
X = vertcat(X{1},X{2});
end
