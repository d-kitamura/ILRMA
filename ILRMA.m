function [Y, cost, W] = ILRMA(X, type, It, nb, drawConv, normalize, W, T, V, Z)
% Independent low-rank matrix analysis (ILRMA)
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% Copyright 2018 Daichi Kitamura
%
% These programs are distributed only for academic research at
% universities and research institutions.
% It is not allowed to use or modify these programs for commercial or
% industrial purpose without our permission.
% When you use or modify these programs and write research articles,
% cite the following references:
%
% # Original paper (The algorithm was called "Rank-1 MNMF" in this paper)
% D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined
% blind source separation unifying independent vector analysis and
% nonnegative matrix factorization," IEEE/ACM Trans. ASLP, vol. 24,
% no. 9, pp. 1626-1641, September 2016.
%
% # Book chapter (The algorithm was renamed as "ILRMA")
% D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined
% blind source separation with independent low-rank matrix analysis,"
% Audio Source Separation. Signals and Communication Technology.,
% S. Makino, Ed. Springer, Cham, pp. 125-155, March 2018.
%
% See also:
% http://d-kitamura.net
% http://d-kitamura.net/demo-ILRMA_en.html
%
% [syntax]
%   [Y, cost, W] = ILRMA(X)
%   [Y, cost, W] = ILRMA(X, type)
%   [Y, cost, W] = ILRMA(X, type, It)
%   [Y, cost, W] = ILRMA(X, type, It, nb)
%   [Y, cost, W] = ILRMA(X, type, It, nb, drawConv)
%   [Y, cost, W] = ILRMA(X, type, It, nb, drawConv, normalize)
%   [Y, cost, W] = ILRMA(X, type, It, nb, drawConv, normalize, W)
%   [Y, cost, W] = ILRMA(X, type, It, nb, drawConv, normalize, W, T)
%   [Y, cost, W] = ILRMA(X, type, It, nb, drawConv, normalize, W, T, V)
%   [Y, cost, W] = ILRMA(X, type, It, nb, drawConv, normalize, W, T, V, Z)
%
% [inputs]
%          X: input multichannel signals in time-frequency domain (frequency bin x time frame x channel)
%       type: 1 or 2 (1: ILRMA without partitioning function (ILRMA1), 2: ILRMA with partitioning function (ILRMA2), default: 1)
%         It: number of iterations (default: 100)
%         nb: number of bases for each source in ILRMA1, or number of bases for all the sources in ILRMA2 (default: time frames/10)
%   drawConv: calculate values of cost function in each iteration for drawing convergence curve or not (true or false, default: false)
%  normalize: normalize variables in each iteration to avoid numerical divergence or not (true or false, default: true, normalization may collapse monotonic decrease of the cost function)
%          W: initial demixing matrix (source x channel x frequency bin, default: identity matrices)
%          T: initial basis matrix (frequency bin x basis x source in ILRMA1, frequency bin x basis for ILRMA2, default: uniform random matrices)
%          V: initial activation matrix (basis x time frame x source in ILRMA1, basis x time frame for ILRMA2, default: uniform random matrices)
%          Z: initial partitioning function (source x basis for ILRMA2, default: uniform random matrices in the range [0,1])
%
% [outputs]
%          Y: estimated multisource signals in time-frequency domain (frequency bin x time frame x source)
%       cost: values of cost function in each iteration (It+1 x 1)
%          W: demixing matrices (source x channel x frequency bin)
%

% Check errors and set default values
[I,J,M] = size(X);
N = M;
if (N > I)
    error('The input spectrogram might be wrong. The size of it must be (freq x frame x ch).\n');
end
if (nargin < 2)
    type = 1; % use ILRMA1 (fixed number of bases for each source)
end
if (nargin < 3)
    It = 100;
end
if (nargin < 4)
    nb = ceil(J/10);
end
if (type == 1)
    L = nb; % number of bases for each source in ILRMA1
elseif (type == 2)
    K = nb; % number of bases for all the sources in ILRMA2
else
    error('The input argument "type" must be 1 or 2.\n');
end
if (nargin < 5)
    drawConv = false;
end
if (nargin < 6)
    normalize = true;
end
if (nargin < 7)
    W = zeros(N,M,I);
    for i=1:I
        W(:,:,i) = eye(N); % initial demixing matrices (identity matrices)
    end
end
if (nargin < 8)
    if (type == 1)
        T = max( rand( I, L, N ), eps ); % initial basis matrix in ILRMA1
    elseif (type == 2)
        T = max( rand( I, K ), eps ); % initial basis matrix in ILRMA2
    end
end
if (nargin < 9)
    if (type == 1)
        V = max( rand( L, J, N ), eps ); % initial activation matrix in ILRMA1
    elseif (type == 2)
        V = max( rand( K, J ), eps ); % initial activation matrix in ILRMA2
    end
end
if (nargin < 10)
    if (type == 2)
        Z = max( rand( N, K ), eps ); % initial partitioning function in ILRMA2
    end
end
if (nargin == 10)
    if (type == 1)
        error('Partitioning function is not required for ILRMA1.\n');
    end
end
if size(W,1) ~= N || size(W,2) ~= M || size(W,3) ~= I
    error('The size of input initial W is incorrect.\n');
end
if (type == 1)
    if (size(T,1) ~= I || size(T,2) ~= L || size(V,1) ~= L || size(V,2) ~= J)
        error('The sizes of input initial T and V are incorrect.\n');
    end
else
    if (size(T,1) ~= I || size(T,2) ~= K || size(V,1) ~= K || size(V,2) ~= J)
        error('The sizes of input initial T and V are incorrect.\n');
    end
end

% Initialization
R = zeros(I,J,N);
Y = zeros(I,J,N);
for i=1:I
    Y(i,:,:) = (W(:,:,i)*squeeze(X(i,:,:)).').'; % initial estimated signals
end
P = max(abs(Y).^2,eps);
UNJI = ones(N,J,I);
Xp = permute(X,[3,2,1]); % M x J x I
cost = zeros(It+1,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%% ILRMA1 %%%%%%%%%%%%%%%%%%%%%%%%%%%
if (type==1) % Algorithm for ILRMA1
    for n=1:N
        R(:,:,n) = T(:,:,n)*V(:,:,n); % low-rank source model
    end
    if drawConv
        cost(1,1) = local_costFunction( P, R, W, I, J );
    end
    % Iterative update
    fprintf('Iteration:    ');
    for it=1:It
        fprintf('\b\b\b\b%4d', it);
        for n=1:N
            %%%%% Update T %%%%%
            T(:,:,n) = T(:,:,n) .* sqrt( (P(:,:,n).*(R(:,:,n).^(-2)))*V(:,:,n).' ./ ( (R(:,:,n).^(-1))*V(:,:,n).' ) );
            T(:,:,n) = max(T(:,:,n),eps);
            R(:,:,n) = T(:,:,n)*V(:,:,n);
            %%%%% Update V %%%%%
            V(:,:,n) = V(:,:,n) .* sqrt( T(:,:,n).'*(P(:,:,n).*(R(:,:,n).^(-2))) ./ ( T(:,:,n).'*(R(:,:,n).^(-1)) ) );
            V(:,:,n) = max(V(:,:,n),eps);
            R(:,:,n) = T(:,:,n)*V(:,:,n);
            %%%%% Update W %%%%%
            Rp = permute(R,[4,2,1,3]); % 1 x J x I x N
            eleinvRp = bsxfun(@rdivide,UNJI,Rp(:,:,:,n)); % N x J x I
            XpHt = conj( permute( Xp, [2,1,3] ) ); % J x M x I (matrix-wise Hermitian transpose)
            D = local_multiplication( Xp.*eleinvRp, XpHt )/J; % M x M x I
            WD = local_multiplicationSquare( W, D, I, M ); % M x M x I
            invWDE = local_inverse( WD, I, M, n ); % M x I
            w = local_wNormalize( invWDE, D, I, M ); % I x M
            W(n,:,:) = w';
            for i=1:I
                Y(i,:,n) = W(n,:,i)*Xp(:,:,i);
            end
        end
        P = max(abs(Y).^2,eps);
        %%%%% Normalization %%%%%
        if normalize
            %%%%%%%%%%%%%%%%%%%%%%%%%%% !!!NOTE!!! %%%%%%%%%%%%%%%%%%%%%%%%
            % This normalization increases the computational stability,   %
            % but the monotonic decrease of the cost function may be lost %
            % because of the numerical errors in this normalization.      %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            lambda = sqrt(sum(sum(P,1),2)/(I*J)); % 1 x 1 x N
            W = bsxfun(@rdivide,W,squeeze(lambda)); % N x M x I
            lambdaPow = lambda.^2; % 1 x 1 x N
            P = bsxfun(@rdivide,P,lambdaPow); % I x J x N
            R = bsxfun(@rdivide,R,lambdaPow); % I x J x N
            T = bsxfun(@rdivide,T,lambdaPow); % I x L x N
        end
        if drawConv
            cost(it+1,1) = local_costFunction( P, R, W, I, J );
        end
    end
    fprintf(' ILRMA1 done.\n');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%% ILRMA2 %%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif (type==2) % Algorithm for ILRMA2
    A = zeros(N,K);
    B = zeros(I,K);
    C = zeros(K,J);
    UN1 = ones(N,1);
    UNN = ones(N,N);
    UI1 = ones(I,1);
    UJ1 = ones(J,1);
    Z  = Z./(UNN*Z); % ensuring sum_n z_{nk}=1
    for n=1:N
        R(:,:,n) = ((UI1)*Z(n,:).*T)*V; % low-rank source model
    end
    if drawConv
        cost(1,1) = local_costFunction( P, R, W, I, J );
    end
    
    % Iterative update
    fprintf('Iteration:    ');
    for it=1:It
        fprintf('\b\b\b\b%4d', it);
        %%%%% Update Z %%%%%
        for n=1:N
            Pn = P(:,:,n); % I x J
            Rn = R(:,:,n); % I x J
            A(n,:) = (( ((T.'*(Pn.*Rn.^(-2))).*V)*UJ1 )./( ((T.'*(Rn.^(-1))).*V)*UJ1 )).';
        end
        Z = Z.*sqrt(A);
        Z = Z./(UNN*Z); % ensuring sum_n z_{nk}=1
        Z = max(Z,eps);
        for n=1:N
            R(:,:,n) = ((UI1)*Z(n,:).*T)*V; % low-rank source model
        end
        %%%%% Update T %%%%%
        for i=1:I
            Pi = squeeze(P(i,:,:)); % J x N
            Ri = squeeze(R(i,:,:)); % J x N
            B(i,:) = (( ((V*(Pi.*Ri.^(-2))).*(Z.'))*UN1 )./( ((V*(Ri.^(-1))).*(Z.'))*UN1 )).';
        end
        T = T.*sqrt(B);
        T = max(T,eps);
        for n=1:N
            R(:,:,n) = ((UI1)*Z(n,:).*T)*V; % low-rank source model
        end
        %%%%% Update V %%%%%
        for j=1:J
            Pj = squeeze(P(:,j,:)); % I x N
            Rj = squeeze(R(:,j,:)); % I x N
            C(:,j) = ( ((T.'*(Pj.*Rj.^(-2))).*(Z.'))*UN1 )./( ((T.'*(Rj.^(-1))).*(Z.'))*UN1 );
        end
        V = V.*sqrt(C);
        V = max(V,eps);
        for n=1:N
            R(:,:,n) = ((UI1)*Z(n,:).*T)*V; % low-rank source model
        end
        %%%%% Update W %%%%%
        for n=1:N
            Rp = permute( R, [4,2,1,3] ); % 1 x J x I x N
            eleinvRp = bsxfun( @rdivide, UNJI, Rp(:,:,:,n) ); % N x J x I
            XpHt = conj( permute( Xp, [2,1,3] ) ); % J x M x I (matrix-wise Hermitian transpose)
            D = local_multiplication( Xp.*eleinvRp, XpHt )/J; % M x M x I
            WD = local_multiplicationSquare( W, D, I, M ); % M x M x I
            invWDE = local_inverse( WD, I, M, n ); % M x I
            w = local_wNormalize( invWDE, D, I, M ); % I x M
            W(n,:,:) = w';
            for i=1:I
                Y(i,:,n) = W(n,:,i)*Xp(:,:,i);
            end
        end
        P = max(abs(Y).^2,eps);
        %%%%% Normalization %%%%%
        if normalize
            %%%%%%%%%%%%%%%%%%%%%%%%%%% !!!NOTE!!! %%%%%%%%%%%%%%%%%%%%%%%%
            % This normalization increases the computational stability,   %
            % but the monotonic decrease of the cost function may be lost %
            % because of the numerical errors in this normalization.      %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            lambda = sqrt(sum(sum(P,1),2)/(I*J)); % 1 x 1 x N
            W = bsxfun(@rdivide,W,squeeze(lambda)); % N x M x I
            lambdaPow = lambda.^2; % 1 x 1 x N
            P = bsxfun(@rdivide,P,lambdaPow); % I x J x N
            R = bsxfun(@rdivide,R,lambdaPow); % I x J x N
            Zlambda = bsxfun(@rdivide,Z,squeeze(lambdaPow)); % N x K
            ZlambdaSum = sum(Zlambda,1); % 1 x K
            T = bsxfun(@times,T,ZlambdaSum); % I x K
            Z = bsxfun(@rdivide,Zlambda,ZlambdaSum); % N x K
        end
        if drawConv
            cost(it+1,1) = local_costFunction( P, R, W, I, J );
        end
    end
    fprintf(' ILRMA2 done.\n');
end

if drawConv
    figure;
    plot( (0:it), cost );
    set(gca,'FontName','Times','FontSize',16);
    xlabel('Iteration','FontName','Arial','FontSize',16);
    ylabel('Value of cost function','FontName','Arial','FontSize',16);
end
end

%% Local functions
%%% Cost function %%%
function [ cost ] = local_costFunction( P, R, W, I, J )
A = zeros(I,1);
for i=1:I
    x = abs(det(W(:,:,i)));
    if x < eps
        x = eps;
    end
    A(i) = log(x);
end
cost = sum(sum(sum(P./R+log(R),3),2),1) - 2*J*sum(A);
end

%%% Multiplication %%%
function [ XY ] = local_multiplication( X, Y )
[A,~,I] = size(X);
[~,B,I] = size(Y);
XY = zeros( A, B, I );
for i = 1:I
    XY(:,:,i) = X(:,:,i)*Y(:,:,i);
end
end

%%% Multiplication of square matrices %%%
function [ XY ] = local_multiplicationSquare( X, Y, I, M )
if M == 2
    XY = zeros( M, M, I );
    XY(1,1,:) = X(1,1,:).*Y(1,1,:) + X(1,2,:).*Y(2,1,:);
    XY(1,2,:) = X(1,1,:).*Y(1,2,:) + X(1,2,:).*Y(2,2,:);
    XY(2,1,:) = X(2,1,:).*Y(1,1,:) + X(2,2,:).*Y(2,1,:);
    XY(2,2,:) = X(2,1,:).*Y(1,2,:) + X(2,2,:).*Y(2,2,:);
elseif M == 3
    XY = zeros( M, M, I );
    XY(1,1,:) = X(1,1,:).*Y(1,1,:) + X(1,2,:).*Y(2,1,:) + X(1,3,:).*Y(3,1,:);
    XY(1,2,:) = X(1,1,:).*Y(1,2,:) + X(1,2,:).*Y(2,2,:) + X(1,3,:).*Y(3,2,:);
    XY(1,3,:) = X(1,1,:).*Y(1,3,:) + X(1,2,:).*Y(2,3,:) + X(1,3,:).*Y(3,3,:);
    XY(2,1,:) = X(2,1,:).*Y(1,1,:) + X(2,2,:).*Y(2,1,:) + X(2,3,:).*Y(3,1,:);
    XY(2,2,:) = X(2,1,:).*Y(1,2,:) + X(2,2,:).*Y(2,2,:) + X(2,3,:).*Y(3,2,:);
    XY(2,3,:) = X(2,1,:).*Y(1,3,:) + X(2,2,:).*Y(2,3,:) + X(2,3,:).*Y(3,3,:);
    XY(3,1,:) = X(3,1,:).*Y(1,1,:) + X(3,2,:).*Y(2,1,:) + X(3,3,:).*Y(3,1,:);
    XY(3,2,:) = X(3,1,:).*Y(1,2,:) + X(3,2,:).*Y(2,2,:) + X(3,3,:).*Y(3,2,:);
    XY(3,3,:) = X(3,1,:).*Y(1,3,:) + X(3,2,:).*Y(2,3,:) + X(3,3,:).*Y(3,3,:);
elseif M == 4
    XY = zeros( M, M, I );
    XY(1,1,:) = X(1,1,:).*Y(1,1,:) + X(1,2,:).*Y(2,1,:) + X(1,3,:).*Y(3,1,:) + X(1,4,:).*Y(4,1,:);
    XY(1,2,:) = X(1,1,:).*Y(1,2,:) + X(1,2,:).*Y(2,2,:) + X(1,3,:).*Y(3,2,:) + X(1,4,:).*Y(4,2,:);
    XY(1,3,:) = X(1,1,:).*Y(1,3,:) + X(1,2,:).*Y(2,3,:) + X(1,3,:).*Y(3,3,:) + X(1,4,:).*Y(4,3,:);
    XY(1,4,:) = X(1,1,:).*Y(1,4,:) + X(1,2,:).*Y(2,4,:) + X(1,3,:).*Y(3,4,:) + X(1,4,:).*Y(4,4,:);
    XY(2,1,:) = X(2,1,:).*Y(1,1,:) + X(2,2,:).*Y(2,1,:) + X(2,3,:).*Y(3,1,:) + X(2,4,:).*Y(4,1,:);
    XY(2,2,:) = X(2,1,:).*Y(1,2,:) + X(2,2,:).*Y(2,2,:) + X(2,3,:).*Y(3,2,:) + X(2,4,:).*Y(4,2,:);
    XY(2,3,:) = X(2,1,:).*Y(1,3,:) + X(2,2,:).*Y(2,3,:) + X(2,3,:).*Y(3,3,:) + X(2,4,:).*Y(4,3,:);
    XY(2,4,:) = X(2,1,:).*Y(1,4,:) + X(2,2,:).*Y(2,4,:) + X(2,3,:).*Y(3,4,:) + X(2,4,:).*Y(4,4,:);
    XY(3,1,:) = X(3,1,:).*Y(1,1,:) + X(3,2,:).*Y(2,1,:) + X(3,3,:).*Y(3,1,:) + X(3,4,:).*Y(4,1,:);
    XY(3,2,:) = X(3,1,:).*Y(1,2,:) + X(3,2,:).*Y(2,2,:) + X(3,3,:).*Y(3,2,:) + X(3,4,:).*Y(4,2,:);
    XY(3,3,:) = X(3,1,:).*Y(1,3,:) + X(3,2,:).*Y(2,3,:) + X(3,3,:).*Y(3,3,:) + X(3,4,:).*Y(4,3,:);
    XY(3,4,:) = X(3,1,:).*Y(1,4,:) + X(3,2,:).*Y(2,4,:) + X(3,3,:).*Y(3,4,:) + X(3,4,:).*Y(4,4,:);
    XY(4,1,:) = X(4,1,:).*Y(1,1,:) + X(4,2,:).*Y(2,1,:) + X(4,3,:).*Y(3,1,:) + X(4,4,:).*Y(4,1,:);
    XY(4,2,:) = X(4,1,:).*Y(1,2,:) + X(4,2,:).*Y(2,2,:) + X(4,3,:).*Y(3,2,:) + X(4,4,:).*Y(4,2,:);
    XY(4,3,:) = X(4,1,:).*Y(1,3,:) + X(4,2,:).*Y(2,3,:) + X(4,3,:).*Y(3,3,:) + X(4,4,:).*Y(4,3,:);
    XY(4,4,:) = X(4,1,:).*Y(1,4,:) + X(4,2,:).*Y(2,4,:) + X(4,3,:).*Y(3,4,:) + X(4,4,:).*Y(4,4,:);
else % slow
    XY = zeros( M, M, I );
    for i = 1:I
        XY(:,:,i) = X(:,:,i)*Y(:,:,i);
    end
end
end

%%% Inverse %%%
function [ invXE ] = local_inverse( X, I, M, n )
if M == 2
    invX = zeros(M,M,I);
    detX = max(X(1,1,:).*X(2,2,:) - X(1,2,:).*X(2,1,:), eps);
    invX(1,1,:) = X(2,2,:);
    invX(1,2,:) = -1*X(1,2,:);
    invX(2,1,:) = -1*X(2,1,:);
    invX(2,2,:) = X(1,1,:);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
    invXE = squeeze(invX(:,n,:)); % multiplying one-hot vector e from right side of invX
elseif M == 3
    invX = zeros(M,M,I);
    detX = max(X(1,1,:).*X(2,2,:).*X(3,3,:) + X(2,1,:).*X(3,2,:).*X(1,3,:) + X(3,1,:).*X(1,2,:).*X(2,3,:) - X(1,1,:).*X(3,2,:).*X(2,3,:) - X(3,1,:).*X(2,2,:).*X(1,3,:) - X(2,1,:).*X(1,2,:).*X(3,3,:), eps);
    invX(1,1,:) = X(2,2,:).*X(3,3,:) - X(2,3,:).*X(3,2,:);
    invX(1,2,:) = X(1,3,:).*X(3,2,:) - X(1,2,:).*X(3,3,:);
    invX(1,3,:) = X(1,2,:).*X(2,3,:) - X(1,3,:).*X(2,2,:);
    invX(2,1,:) = X(2,3,:).*X(3,1,:) - X(2,1,:).*X(3,3,:);
    invX(2,2,:) = X(1,1,:).*X(3,3,:) - X(1,3,:).*X(3,1,:);
    invX(2,3,:) = X(1,3,:).*X(2,1,:) - X(1,1,:).*X(2,3,:);
    invX(3,1,:) = X(2,1,:).*X(3,2,:) - X(2,2,:).*X(3,1,:);
    invX(3,2,:) = X(1,2,:).*X(3,1,:) - X(1,1,:).*X(3,2,:);
    invX(3,3,:) = X(1,1,:).*X(2,2,:) - X(1,2,:).*X(2,1,:);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
    invXE = squeeze(invX(:,n,:)); % multiplying one-hot vector e from right side of invX
elseif M == 4
    invX = zeros(M,M,I);
    detX = max(X(1,1,:).*X(2,2,:).*X(3,3,:).*X(4,4,:) + X(1,1,:).*X(2,3,:).*X(3,4,:).*X(4,2,:) + X(1,1,:).*X(2,4,:).*X(3,2,:).*X(4,3,:) + X(1,2,:).*X(2,1,:).*X(3,4,:).*X(4,3,:) + X(1,2,:).*X(2,3,:).*X(3,1,:).*X(4,4,:) + X(1,2,:).*X(2,4,:).*X(3,3,:).*X(4,1,:) + X(1,3,:).*X(2,1,:).*X(3,2,:).*X(4,4,:) + X(1,3,:).*X(2,2,:).*X(3,4,:).*X(4,1,:) + X(1,3,:).*X(2,4,:).*X(3,1,:).*X(4,2,:) + X(1,4,:).*X(2,1,:).*X(3,3,:).*X(4,2,:) + X(1,4,:).*X(2,2,:).*X(3,1,:).*X(4,3,:) + X(1,4,:).*X(2,3,:).*X(3,2,:).*X(4,1,:) - X(1,1,:).*X(2,2,:).*X(3,4,:).*X(4,3,:) - X(1,1,:).*X(2,3,:).*X(3,2,:).*X(4,4,:) - X(1,1,:).*X(2,4,:).*X(3,3,:).*X(4,2,:) - X(1,2,:).*X(2,1,:).*X(3,3,:).*X(4,4,:) - X(1,2,:).*X(2,3,:).*X(3,4,:).*X(4,1,:) - X(1,2,:).*X(2,4,:).*X(3,1,:).*X(4,3,:) - X(1,3,:).*X(2,1,:).*X(3,4,:).*X(4,2,:) - X(1,3,:).*X(2,2,:).*X(3,1,:).*X(4,4,:) - X(1,3,:).*X(2,4,:).*X(3,2,:).*X(4,1,:) - X(1,4,:).*X(2,1,:).*X(3,2,:).*X(4,3,:) - X(1,4,:).*X(2,2,:).*X(3,3,:).*X(4,1,:) - X(1,4,:).*X(2,3,:).*X(3,1,:).*X(4,2,:), eps);
    invX(1,1,:) = X(2,2,:).*X(3,3,:).*X(4,4,:) + X(2,3,:).*X(3,4,:).*X(4,2,:) + X(2,4,:).*X(3,2,:).*X(4,3,:) - X(2,2,:).*X(3,4,:).*X(4,3,:) - X(2,3,:).*X(3,2,:).*X(4,4,:) - X(2,4,:).*X(3,3,:).*X(4,2,:);
    invX(1,2,:) = X(1,2,:).*X(3,4,:).*X(4,3,:) + X(1,3,:).*X(3,2,:).*X(4,4,:) + X(1,4,:).*X(3,3,:).*X(4,2,:) - X(1,2,:).*X(3,3,:).*X(4,4,:) - X(1,3,:).*X(3,4,:).*X(4,2,:) - X(1,4,:).*X(3,2,:).*X(4,3,:);
    invX(1,3,:) = X(1,2,:).*X(2,3,:).*X(4,4,:) + X(1,3,:).*X(2,4,:).*X(4,2,:) + X(1,4,:).*X(2,2,:).*X(4,3,:) - X(1,2,:).*X(2,4,:).*X(4,3,:) - X(1,3,:).*X(2,2,:).*X(4,4,:) - X(1,4,:).*X(2,3,:).*X(4,2,:);
    invX(1,4,:) = X(1,2,:).*X(2,4,:).*X(3,3,:) + X(1,3,:).*X(2,2,:).*X(3,4,:) + X(1,4,:).*X(2,3,:).*X(3,2,:) - X(1,2,:).*X(2,3,:).*X(3,4,:) - X(1,3,:).*X(2,4,:).*X(3,2,:) - X(1,4,:).*X(2,2,:).*X(3,3,:);
    invX(2,1,:) = X(2,1,:).*X(3,4,:).*X(4,3,:) + X(2,3,:).*X(3,1,:).*X(4,4,:) + X(2,4,:).*X(3,3,:).*X(4,1,:) - X(2,1,:).*X(3,3,:).*X(4,4,:) - X(2,3,:).*X(3,4,:).*X(4,1,:) - X(2,4,:).*X(3,1,:).*X(4,3,:);
    invX(2,2,:) = X(1,1,:).*X(3,3,:).*X(4,4,:) + X(1,3,:).*X(3,4,:).*X(4,1,:) + X(1,4,:).*X(3,1,:).*X(4,3,:) - X(1,1,:).*X(3,4,:).*X(4,3,:) - X(1,3,:).*X(3,1,:).*X(4,4,:) - X(1,4,:).*X(3,3,:).*X(4,1,:);
    invX(2,3,:) = X(1,1,:).*X(2,4,:).*X(4,3,:) + X(1,3,:).*X(2,1,:).*X(4,4,:) + X(1,4,:).*X(2,3,:).*X(4,1,:) - X(1,1,:).*X(2,3,:).*X(4,4,:) - X(1,3,:).*X(2,4,:).*X(4,1,:) - X(1,4,:).*X(2,1,:).*X(4,3,:);
    invX(2,4,:) = X(1,1,:).*X(2,3,:).*X(3,4,:) + X(1,3,:).*X(2,4,:).*X(3,1,:) + X(1,4,:).*X(2,1,:).*X(3,3,:) - X(1,1,:).*X(2,4,:).*X(3,3,:) - X(1,3,:).*X(2,1,:).*X(3,4,:) - X(1,4,:).*X(2,3,:).*X(3,1,:);
    invX(3,1,:) = X(2,1,:).*X(3,2,:).*X(4,4,:) + X(2,2,:).*X(3,4,:).*X(4,1,:) + X(2,4,:).*X(3,1,:).*X(4,2,:) - X(2,1,:).*X(3,4,:).*X(4,2,:) - X(2,2,:).*X(3,1,:).*X(4,4,:) - X(2,4,:).*X(3,2,:).*X(4,1,:);
    invX(3,2,:) = X(1,1,:).*X(3,4,:).*X(4,2,:) + X(1,2,:).*X(3,1,:).*X(4,4,:) + X(1,4,:).*X(3,2,:).*X(4,1,:) - X(1,1,:).*X(3,2,:).*X(4,4,:) - X(1,2,:).*X(3,4,:).*X(4,1,:) - X(1,4,:).*X(3,1,:).*X(4,2,:);
    invX(3,3,:) = X(1,1,:).*X(2,2,:).*X(4,4,:) + X(1,2,:).*X(2,4,:).*X(4,1,:) + X(1,4,:).*X(2,1,:).*X(4,2,:) - X(1,1,:).*X(2,4,:).*X(4,2,:) - X(1,2,:).*X(2,1,:).*X(4,4,:) - X(1,4,:).*X(2,2,:).*X(4,1,:);
    invX(3,4,:) = X(1,1,:).*X(2,4,:).*X(3,2,:) + X(1,2,:).*X(2,1,:).*X(3,4,:) + X(1,4,:).*X(2,2,:).*X(3,1,:) - X(1,1,:).*X(2,2,:).*X(3,4,:) - X(1,2,:).*X(2,4,:).*X(3,1,:) - X(1,4,:).*X(2,1,:).*X(3,2,:);
    invX(4,1,:) = X(2,1,:).*X(3,3,:).*X(4,2,:) + X(2,2,:).*X(3,1,:).*X(4,3,:) + X(2,3,:).*X(3,2,:).*X(4,1,:) - X(2,1,:).*X(3,2,:).*X(4,3,:) - X(2,2,:).*X(3,3,:).*X(4,1,:) - X(2,3,:).*X(3,1,:).*X(4,2,:);
    invX(4,2,:) = X(1,1,:).*X(3,2,:).*X(4,3,:) + X(1,2,:).*X(3,3,:).*X(4,1,:) + X(1,3,:).*X(3,1,:).*X(4,2,:) - X(1,1,:).*X(3,3,:).*X(4,2,:) - X(1,2,:).*X(3,1,:).*X(4,3,:) - X(1,3,:).*X(3,2,:).*X(4,1,:);
    invX(4,3,:) = X(1,1,:).*X(2,3,:).*X(4,2,:) + X(1,2,:).*X(2,1,:).*X(4,3,:) + X(1,3,:).*X(2,2,:).*X(4,1,:) - X(1,1,:).*X(2,2,:).*X(4,3,:) - X(1,2,:).*X(2,3,:).*X(4,1,:) - X(1,3,:).*X(2,1,:).*X(4,2,:);
    invX(4,4,:) = X(1,1,:).*X(2,2,:).*X(3,3,:) + X(1,2,:).*X(2,3,:).*X(3,1,:) + X(1,3,:).*X(2,1,:).*X(3,2,:) - X(1,1,:).*X(2,3,:).*X(3,2,:) - X(1,2,:).*X(2,1,:).*X(3,3,:) - X(1,3,:).*X(2,2,:).*X(3,1,:);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
    invXE = squeeze(invX(:,n,:)); % multiplying one-hot vector e from right side of invX
else % slow
    eyeM = eye(M);
    row = repmat(reshape(1:M*I,[M,1,I]),[1 M 1]);
    col = repmat(reshape(1:M*I,[1,M,I]),[M 1 1]);
    invX = reshape((sparse(row(:),col(:),X(:))\repmat(eyeM,[I,1])).', [M, M, I]);
    invX = full(permute(invX,[2,1,3]));
    invXE = squeeze(invX(:,n,:)); % multiplying one-hot vector e from right side of invX
end
end

%%% Normalization of separation filter w %%%
function [ w ] = local_wNormalize( w, D, I, M )
if M == 2
    D = permute(D,[3,1,2]); % I x N x N
    wHt = w'; % I x N
    w = w.'; % I x N
    wHtD(:,1) = wHt(:,1).*D(:,1,1) + wHt(:,2).*D(:,2,1);
    wHtD(:,2) = wHt(:,1).*D(:,1,2) + wHt(:,2).*D(:,2,2);
    normCoef = sqrt( wHtD(:,1).*w(:,1) + wHtD(:,2).*w(:,2) );
    w = bsxfun(@rdivide,w,max(normCoef,eps));
elseif M == 3
    D = permute(D,[3,1,2]); % I x N x N
    wHt = w'; % I x N
    w = w.'; % I x N
    wHtD(:,1) = wHt(:,1).*D(:,1,1) + wHt(:,2).*D(:,2,1) + wHt(:,3).*D(:,3,1);
    wHtD(:,2) = wHt(:,1).*D(:,1,2) + wHt(:,2).*D(:,2,2) + wHt(:,3).*D(:,3,2);
    wHtD(:,3) = wHt(:,1).*D(:,1,3) + wHt(:,2).*D(:,2,3) + wHt(:,3).*D(:,3,3);
    normCoef = sqrt( wHtD(:,1).*w(:,1) + wHtD(:,2).*w(:,2) + wHtD(:,3).*w(:,3) );
    w = bsxfun(@rdivide,w,max(normCoef,eps));
elseif M == 4
    D = permute(D,[3,1,2]); % I x N x N
    wHt = w'; % I x N
    w = w.'; % I x N
    wHtD(:,1) = wHt(:,1).*D(:,1,1) + wHt(:,2).*D(:,2,1) + wHt(:,3).*D(:,3,1) + wHt(:,4).*D(:,4,1);
    wHtD(:,2) = wHt(:,1).*D(:,1,2) + wHt(:,2).*D(:,2,2) + wHt(:,3).*D(:,3,2) + wHt(:,4).*D(:,4,2);
    wHtD(:,3) = wHt(:,1).*D(:,1,3) + wHt(:,2).*D(:,2,3) + wHt(:,3).*D(:,3,3) + wHt(:,4).*D(:,4,3);
    wHtD(:,4) = wHt(:,1).*D(:,1,4) + wHt(:,2).*D(:,2,4) + wHt(:,3).*D(:,3,4) + wHt(:,4).*D(:,4,4);
    normCoef = sqrt( wHtD(:,1).*w(:,1) + wHtD(:,2).*w(:,2) + wHtD(:,3).*w(:,3) + wHtD(:,4).*w(:,4) );
    w = bsxfun(@rdivide,w,max(normCoef,eps));
else
    wHt = w'; % I x N
    normCoef = zeros(1,I);
    for i = 1:I
        normCoef(1,i) = wHt(i,:)*D(:,:,i)*w(:,i);
    end
    w = bsxfun(@rdivide,w,max(sqrt(normCoef),eps)).';
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%