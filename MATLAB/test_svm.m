function y = test_svm(X, S, T, model, data_mean, data_std)

global B;
if any(size(B) ~= [length(T),length(S)])
    B = zeros(length(T),length(S));
end
B = [B;X]; B = B(end-length(T)+1:end,:);

X = log(var(T*(B*S')));
%X = (X-data_mean)./data_std;
y = -predict(model, X);
end