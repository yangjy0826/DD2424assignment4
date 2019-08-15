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

