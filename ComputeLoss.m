function [J, h] = ComputeLoss(X, Y, RNN, h0)
b = RNN.b; % m, 1
c = RNN.c; % K, 1
U = RNN.U; % m, K
W = RNN.W; % m, m
V = RNN.V; % K, m
tau = size(X, 2);
ht = h0;
J = 0; 
m = size(b, 1);
K = size(c, 1);
a = zeros(m, tau); % at: m,1
h = zeros(m, tau); % ht: m,1
o = zeros(K, tau); % ot: K,1
p = zeros(K, tau); % pt: K,1
%X,Y size: K, seq_length
for t=1:tau
    at = W*ht+U*X(:, t)+b;
    a(:, t) = at;
    ht = tanh(at);
    h(:, t) = ht;
    ot = V*ht+c;
    o(:, t) = ot;
    pt = softmax(ot);
    p(:, t) = pt;
    J = J - log(Y(:, t)'*pt);
end
h = [h0, h];
end

