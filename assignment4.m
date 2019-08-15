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