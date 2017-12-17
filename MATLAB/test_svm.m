function y = test_svm(X, S, T, model)

global B;
if any(size(B) ~= [length(T),length(S)])
    B = zeros(length(T),length(S));
end
B = [B;X]; B = B(end-length(T)+1:end,:);

X = log(var(T*(B*S')));
y = -predict(model, X);
end