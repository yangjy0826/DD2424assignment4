function [RNN, M] = BackwardPass(X, Y, RNN, M, h, eta)
grad = ComputeGradients(X, Y, RNN, h(:, 1));
%grad = ComputeGradients(X, Y, RNN, h);
for f = fieldnames(RNN)'
    M.(f{1}) = M.(f{1}) + grad.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*(grad.(f{1})./(M.(f{1}) + eps).^(0.5));
end
end

