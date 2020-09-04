function [Y, cost, W] = consistentILRMA(X, It, nb, fftSize, shiftSize, window, mixLen, refMic, drawConv, W, T, V)
% Independent low-rank matrix analysis (ILRMA)
%
% [syntax]
%   [Y, cost, W] = consistentILRMA(X, It, nb, fftSize, shiftSize, window, mix, refMic)
%   [Y, cost, W] = consistentILRMA(X, It, nb, fftSize, shiftSize, window, mix, refMic, W)
%   [Y, cost, W] = consistentILRMA(X, It, nb, fftSize, shiftSize, window, mix, refMic, W, T)
%   [Y, cost, W] = consistentILRMA(X, It, nb, fftSize, shiftSize, window, mix, refMic, W, T, V)
%
% [inputs]
%          X: input multichannel signals in time-frequency domain (frequency bin x time frame x channel)
%         It: number of iterations
%         nb: number of bases for each source in ILRMA1, or number of bases for all the sources in ILRMA2
%    fftSize: fft length in STFT
%  shiftSize: window shift size in STFT
%     window: window function in STFT
%     mixLen: length of mixture signal in waveform domain
%     refMic: reference microphone for projection back
%   drawConv: calculate values of cost function in each iteration for drawing convergence curve or not (true or false, default: false)
%          W: initial demixing matrix (source x channel x frequency bin, default: identity matrices)
%          T: initial basis matrix (frequency bin x basis x source in ILRMA1, frequency bin x basis for ILRMA2, default: uniform random matrices)
%          V: initial activation matrix (basis x time frame x source in ILRMA1, basis x time frame for ILRMA2, default: uniform random matrices)
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
if (nargin < 9)
    drawConv = false;
end
if (nargin < 10)
    W = zeros(N,M,I);
    for i=1:I
        W(:,:,i) = eye(N); % initial demixing matrices (identity matrices)
    end
end
if (nargin < 11)
    T = max( rand( I, nb, N ), eps ); % initial basis matrix in ILRMA1
end
if (nargin < 12)
    V = max( rand( nb, J, N ), eps ); % initial activation matrix in ILRMA1
end
if size(W,1) ~= N || size(W,2) ~= M || size(W,3) ~= I
    error('The size of input initial W is incorrect.\n');
end
if (size(T,1) ~= I || size(T,2) ~= nb || size(V,1) ~= nb || size(V,2) ~= J)
    error('The sizes of input initial T and V are incorrect.\n');
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

%%%%%%%%%%%%%%%%%%%%%%%%%%% Consisten ILRMA %%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    if drawConv
        cost(it+1,1) = local_costFunction( P, R, W, I, J );
    end
    
    % Apply projection back
    lambda = local_projectionBack(Y,X(:,:,refMic)); % N x 1 x I
    W = bsxfun(@times,W,lambda);
    lambdaPow = abs(lambda).^2; % N x 1 x I
    lambdaIJ = permute(lambdaPow,[3,2,1]); % I x 1 x N
    R = bsxfun(@times,R,lambdaIJ);
    lambdaIL = permute(lambdaPow,[3,2,1]); % I x 1 x N
    T = bsxfun(@times,T,lambdaIL);
    
    % Recompute Y after applying projection back
    for i=1:I
        Y(i,:,:) = (W(:,:,i)*squeeze(X(i,:,:)).').';
    end
    
    % Apply ISTFT and STFT for ensuring spectrogram consistency
    Y = STFT( ISTFT(Y, shiftSize, window, mixLen), fftSize, shiftSize, window );
    P = max(abs(Y).^2,eps); % recompute power spectrogram
end
fprintf(' Consisten ILRMA done.\n');

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

%%% Projection back that returns only frequency-wise coefficients %%%
function [ D ] = local_projectionBack( Y, X )
[I,~,N] = size(Y);
A = zeros(1,N,I);
D = zeros(N,1,I);
for i=1:I
    Yi = squeeze(Y(i,:,:)).'; % channels x frames (N x J)
    A(1,:,i) = X(i,:,1)*Yi'/(Yi*Yi');
end
A(isnan(A)) = 0;
A(isinf(A)) = 0;
for n=1:N
    for i=1:I
        D(n,1,i)=A(1,n,i);
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%