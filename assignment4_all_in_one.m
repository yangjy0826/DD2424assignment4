clc;
clear;

% 0.1 Read the data
% Read the contents of the book
book_fname = 'goblet_book.txt';
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);
book_chars = unique(book_data); % Get the unique characters in the book
K = size(book_chars, 2); %  dimensionality of the output (input) vector of this RNN
% Initialize map containers:
char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');
char_set = num2cell(book_chars); 
ind_set = 1:K;
map1 = containers.Map(char_set, ind_set); % keySet:char_set, valueSet:ind_set
map2 = containers.Map(ind_set, char_set); % keySet:ind_set, valueSet:char_set
char_to_ind = [char_to_ind; map1];
ind_to_char = [ind_to_char; map2];

% 0.2  Set hyper-parameters & initialize the RNN¡¯s parameters
m = 100;  % dimensionality of RNN¡¯s hidden state
% gradient check: m=5; train:m=100
eta = 0.1; % learning rate
seq_length = 25; % the length of the input sequences
% bias vectors:
RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
sig = 0.01;
% weight matrices:
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

%% Gradient check
% 0.3 Synthesize text from your randomly initialized RNN
% function: SynthesizeText.m
% 0.4 Implement the forward & backward pass of back-prop
% functions: ComputeLoss, ComputeGradients
% X_chars = book_data(1:seq_length);
% Y_chars = book_data(2:seq_length+1);
% X_ind = zeros(1, seq_length);
% Y_ind = zeros(1, seq_length);
% for i=1:seq_length
%     X_ind(i) = char_to_ind(X_chars(i));
%     Y_ind(i) = char_to_ind(Y_chars(i));
% end
% X = onehot(X_ind, K); % K, seq_length
% Y = onehot(Y_ind, K); % K, seq_length
% h0 = zeros(m, 1);

% h = 1e-4; % the step size for the numerical computations, ComputeGradsNum
% num_grad = ComputeGradsNum(X, Y, RNN, h);
% grad = ComputeGradients(X, Y, RNN, h0);
% eps = 1e-5;
% dif_b = abs(num_grad.b-grad.b)./max(eps,sum(abs(num_grad.b)+ abs(num_grad.b)));
% re_b = max(dif_b);
% dif_c = abs(num_grad.c-grad.c)./max(eps,sum(abs(num_grad.c)+ abs(num_grad.c)));
% re_c = max(dif_c);
% dif_U = abs(num_grad.U-grad.U)./max(eps,sum(abs(num_grad.U)+ abs(num_grad.U)));
% re_U = max(max(dif_U));
% dif_W = abs(num_grad.W-grad.W)./max(eps,sum(abs(num_grad.W)+ abs(num_grad.W)));
% re_W = max(max(dif_W));
% dif_V = abs(num_grad.V-grad.V)./max(eps,sum(abs(num_grad.V)+ abs(num_grad.V)));
% re_V = max(max(dif_V));

%% training
% 0.5 Train your RNN using AdaGrad
M.W = zeros(size(RNN.W));
M.U = zeros(size(RNN.U));
M.V = zeros(size(RNN.V));
M.b = zeros(size(RNN.b));
M.c = zeros(size(RNN.c));

n = size(book_data, 2);
X_chars = book_data(1:n);
X_ind = zeros(1, n);
for i=1:n
    X_ind(i) = char_to_ind(X_chars(i));
end
X = onehot(X_ind, K); % K, n
Y = X; % K, n
h0 = zeros(m, 1);
% Gradient descent parameters:
GDpara.eta = 0.1;
GDpara.n_epochs = 8;
GDpara.iter = 1;
min.loss = 1000;
sl = 0;
SL = [];
for i = 1:GDpara.n_epochs
    [RNN, M, GDpara, min, sl] = MiniBatchGD(RNN, M, X, Y, seq_length, n, GDpara, min, ind_to_char, sl(end));
    SL = [SL sl];
end
% smoothloss is not inserted into minibatchGD
% the SL why in the loop below will updqate again (already update once in the minibtach)
plot(SL);
grid on;
xlabel('iter');
ylabel('smooth loss');

%% Functions
function [RNN, M] = BackwardPass(X, Y, RNN, M, h, eta)
grad = ComputeGradients(X, Y, RNN, h(:, 1));
%grad = ComputeGradients(X, Y, RNN, h);
for f = fieldnames(RNN)'
    M.(f{1}) = M.(f{1}) + grad.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*(grad.(f{1})./(M.(f{1}) + eps).^(0.5));
end
end

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

function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end

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

function [RNN, M, GDpara, min, sl] = MiniBatchGD(RNN, M, X, Y, seq_length, n, GDpara, min, ind_to_char, smooth_loss)
e = 1;
tau = seq_length;
b = RNN.b; % m, 1
% c = RNN.c; % K, 1
% U = RNN.U; % m, K
% W = RNN.W; % m, m
% V = RNN.V; % K, m
m = size(b, 1);
% K = size(c, 1);
sl = []; % a set of smooth loss

while e <= n - tau - 1 % n = size(book_data, 2);
    if e == 1
        hprev = zeros(m, 1);
    else
        hprev = h(:, end); % e>1 then hprev should be set to the last computed hidden state by the forward pass in the previous iteration
    end
    Xe = X(:, e:e+tau-1);
    Ye = Y(:, e+1:e+tau);
    
    % ForwardPass:
    %X,Y size: K, tau
    ht = hprev; 
    [J, h] = ComputeLoss(Xe, Ye, RNN, ht);
%     a = zeros(m, tau); % at: m,1
%     h = zeros(m, tau); % ht: m,1
%     o = zeros(K, tau); % ot: K,1
%     p = zeros(K, tau); % pt: K,1
%     J = 0; 
%     for t=1:tau
%         at = W*ht+U*Xe(:, t)+b;
%         a(:, t) = at;
%         ht = tanh(at);
%         h(:, t) = ht;
%         ot = V*ht+c;
%         o(:, t) = ot;
%         pt = softmax(ot);
%         p(:, t) = pt;
%         J = J - log(Ye(:, t)'*p(:, t)); %loss
%     end
%     h = [hprev, h];

    [RNN, M] = BackwardPass(Xe, Ye, RNN, M, h, GDpara.eta);
    
    if GDpara.iter == 1 && e == 1
        smooth_loss = J;
    end
    smooth_loss = 0.999*smooth_loss + 0.001*J;
    if smooth_loss < min.loss
        min.RNN = RNN;
        min.h = hprev;
        min.iter = GDpara.iter;
        min.J = smooth_loss;
    end
    
    sl = [sl, smooth_loss];
    if GDpara.iter == 1 ||  mod(GDpara.iter, 500)==0
        y = SynthesizeText(hprev, Xe(:, 1), RNN, 1000);
        y_char = [];
        for i = 1 : 1000
            y_char = [y_char ind_to_char(y(i))];
        end
        fprintf('iter = %d, smooth_loss = %.3f, \n seq = %s\n\n',GDpara.iter, smooth_loss, y_char);
    end
    
    GDpara.iter= GDpara.iter+1;
    e= e+ tau;
end
end

function out = onehot(label, K)
% DENOTE N as the number of images, K as the number of label kinds
% INPUT     - label:        1*N
% OUTPUT    - out:          K*N
N = length(label);
out = zeros(K, N);
for i = 1 : N
    out(label(i), i) = 1;
end
end

function y = SynthesizeText(h0, x0, RNN, n)
b = RNN.b; % m, 1
c = RNN.c; % K, 1
U = RNN.U; % m, K
W = RNN.W; % m, m
V = RNN.V; % K, m
h = h0;
x = x0;
y = zeros(1, n);
% Similar to ComputeCost function, but with one difference,
% the xnext need to be generated by x
for i=1:n % n: the length of the text
    a = W*h+U*x+b;
    h = tanh(a);
    o = V*h+c;
    p = softmax(o);
    %  randomly select a character based on the output probability scores p:
    cp = cumsum(p); % compute the vector containing the cumulative sum of the probabilities
    a = rand; % generate a random draw, a, from a uniform distribution in the range 0 to 1
    ixs = find(cp-a >0);
    ii = ixs(1);
    K = size(c, 1);
    % Y is the one-hot encoding of each sampled character:
    x = onehot(ii, K);
    y(i) = ii;
end