function grad = ComputeGradients(X, Y, RNN, h0)
b = RNN.b; % m, 1
c = RNN.c; % K, 1
U = RNN.U; % m, K
W = RNN.W; % m, m
V = RNN.V; % K, m
tau = size(X, 2); % seq_length
ht = h0;
J = 0; 
m = size(b, 1);
K = size(c, 1);
a = zeros(m, tau); % at: m,1
h = zeros(m, tau); % ht: m,1
o = zeros(K, tau); % ot: K,1
p = zeros(K, tau); % pt: K,1
%X,Y size: K, tau
for t=1:tau
    at = W*ht+U*X(:, t)+b;
    a(:, t) = at;
    ht = tanh(at);
    h(:, t) = ht;
    ot = V*ht+c;
    o(:, t) = ot;
    pt = softmax(ot);
    p(:, t) = pt;
    J = J - log(Y(:, t)'*p(:, t));
end
h = [h0, h];

g_h = zeros(tau, m);
g_a = zeros(tau, m);

g = -(Y - p)';               % tau,K
grad.c = (sum(g))';          % K,tau
grad.V = g'*h(:, 2 : end)';  % tau,K

g_h(tau, :) = g(tau, :)*V;                                  % 1,m
g_a(tau, :) = g_h(tau, :)*diag(1 - (tanh(a(:, tau))).^2);     % 1,m

for t = tau - 1 : -1 : 1
    g_h(t, :) = g(t, :)*V + g_a(t + 1, :)*W;
    g_a(t, :) = g_h(t, :)*diag(1 - (tanh(a(:, t))).^2);
end

grad.b = (sum(g_a))';                  % m, tau
grad.W = g_a'*h(:, 1 : end - 1)';      % m, m
grad.U = g_a'*X';                      % m, K
end

