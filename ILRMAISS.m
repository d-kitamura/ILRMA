function [estSig, cost] = ILRMAISS(mixSig, nSrc, sampFreq, nBases, fftSize, shiftSize, windowType, nIter, ilrmaType, refMic, applyNormalize, applyWhitening, drawConv)
% Blind source separation with independent low-rank matrix analysis (ILRMA) based on iterative source steering update rule
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% Copyright 2021 Daichi Kitamura
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
% # Original AuxIVA-ISS paper
% S. Robin and N. Ono, 
% "Fast and stable blind source separation with rank-1 updates," 
% Proc. ICASSP, pp.236–240, 2020.
%
% See also:
% http://d-kitamura.net
% http://d-kitamura.net/demo-ILRMA_en.html
%
% [syntax]
%   [estSig,cost] = ILRMAISS(mixSig,nSrc,sampFreq,nBases,fftSize,shiftSize,windowType,nIter,ilrmaType,refMic,applyNormalize,applyWhitening,drawConv)
%
% [inputs]
%         mixSig: observed mixture (sigLen x nCh)
%           nSrc: number of sources in the mixture (scalar)
%       sampFreq: sampling frequency [Hz] of mixSig (scalar)
%         nBases: number of bases in NMF model (scalar, # of bases for "each" source when ilrmaType=1, and # of bases for "all" sources when ilrmaType=2, default: 4)
%        fftSize: window length [points] in STFT (scalar, default: next higher power of 2 that exceeds 0.256*sampFreq)
%      shiftSize: shift length [points] in STFT (scalar, default: fftSize/2)
%     windowType: window function used in STFT (name of window function, default: 'hamming')
%          nIter: number of iterations in the parameter update in ILRMA (scalar, default: 100)
%      ilrmaType: without or with partitioning function (1: ILRMA without partitioning function (ILRMA type1), 2: ILRMA with partitioning function (ILRMA type2), default: 1)
%         refMic: reference microphone for applying back projection (default: 1)
% applyNormalize: normalize parameters in each iteration to avoid numerical divergence (0: do not apply, 1: average-power-based normalization, 2: back projection (only for ILRMA type1), normalization may collapse monotonic decrease of the cost function, default: 0)
% applyWhitening: apply whitening to the observed multichannel spectrograms or not (true or false, default: true)
%       drawConv: plot cost function values in each iteration or not (true or false, default: false)
%
% [outputs]
%         estSig: estimated signals (sigLen x nCh x nSrc)
%           cost: convergence behavior of cost function in ILRMA (nIter+1 x 1)
%

% Arguments check and set default values
arguments
    mixSig (:,:) double
    nSrc (1,1) double {mustBeInteger(nSrc)}
    sampFreq (1,1) double
    nBases (1,1) double {mustBeInteger(nBases)} = 4
    fftSize (1,1) double {mustBeInteger(fftSize)} = 2^nextpow2(0.256*sampFreq)
    shiftSize (1,1) double {mustBeInteger(shiftSize)} = fftSize/2
    windowType char {mustBeMember(windowType,{'hamming','hann','rectangular','blackman','sine'})} = 'hamming'
    nIter (1,1) double {mustBeInteger(nIter)} = 100
    ilrmaType (1,1) double {mustBeInteger(ilrmaType)} = 1
    refMic (1,1) double {mustBeInteger(refMic)} = 1
    applyNormalize (1,1) double {mustBeInteger(applyNormalize)} = 0
    applyWhitening (1,1) logical = true
    drawConv (1,1) logical = false
end

% Error check
[sigLen, nCh] = size(mixSig); % sigLen: signal length, nCh: number of channels
if sigLen < nCh; error("The size of mixSig might be wrong.\n"); end
if nCh < nSrc || nSrc < 2; error("The number of channels must be equal to or grater than the number of sources in the mixture.\n"); end
if sampFreq <= 0; error("The sampling frequency (sampFreq) must be a positive value.\n"); end
if nBases < 1; error("The number of bases (nBases) must be a positive integer value.\n"); end
if fftSize < 1; error("The FFT length in STFT (fftSize) must be a positive integer value.\n"); end
if shiftSize < 1; error("The shift length in STFT (shiftSize) must be a positive integer value.\n"); end
if nIter < 1; error("The number of iterations (nIter) must be a positive integer value.\n"); end
if ilrmaType ~= 1 && ilrmaType ~= 2; error("The ILRMA type (ilrmaType) must be set to 1 or 2.\n"); end
if refMic < 1 || refMic > nCh; error("The reference microphone must be an integer between 1 and nCh.\n"); end
if applyNormalize ~= 0 && applyNormalize ~= 1 && applyNormalize ~= 2; error("The normalization type (applyNormalize) must be set to 0, 1, or 2.\n"); end
if applyNormalize == 2 && ilrmaType == 2; error("The back-projection-based normalization only supports ILRMA type 1.\n"); end

% Apply multichannel short-time Fourier transform (STFT)
[mixSpecgram, windowInStft] = STFT(mixSig, fftSize, shiftSize, windowType);

% Apply whitening (decorrelate X so that the correlation matrix becomes an identity matrix) based on principal component analysis
if applyWhitening
    inputMixSpecgram = whitening(mixSpecgram, nSrc); % apply whitening, where dimension is reduced from nCh to nSrc when nSrc < nCh
else
    inputMixSpecgram = mixSpecgram(:,:,1:nSrc); % when nSrc < nCh, only mixSpecgram(:,:,1:nSrc) is input to ILRMA so that the number of microphones equals to the number of sources (determined condition)
end

% Apply ILRMA
if ilrmaType == 1
    [estSpecgram, cost] = local_ILRMA1ISS(inputMixSpecgram, nIter, nBases, applyNormalize, drawConv, mixSpecgram(:,:,refMic));
else
    [estSpecgram, cost] = local_ILRMA2ISS(inputMixSpecgram, nIter, nBases, applyNormalize, drawConv);
end

% Apply back projection (fix the scale ambiguity using the reference microphone channel)
scaleFixedSepSpecgram = backProjection(estSpecgram, mixSpecgram(:,:,refMic)); % scale-fixed estimated signal

% Inverse STFT for each source
estSig = ISTFT(scaleFixedSepSpecgram, shiftSize, windowInStft, sigLen);
end

%% Local function for ILRMA type1 (without pertitioning function) based on ISS
function [Y, cost] = local_ILRMA1ISS(X, nIter, L, applyNormalize, drawConv, refMixSpecgram)
% [inputs]
%              X: observed multichannel spectrograms (I x J x M)
%          nIter: number of iterations of the parameter updates
%              L: number of bases in NMF model for each source
% applyNormalize: normalize parameters in each iteration to avoid numerical divergence (0: do not apply, 1: average-power-based normalization, 2: back projection, normalization may collapse monotonic decrease of the cost function)
%       drawConv: plot cost function values in each iteration or not (true or false)
% refMixSpecgram: observed reference spectrogram before apply whitening (I x J)
%
% [outputs]
%              Y: estimated spectrograms of sources (I x J x N)
%           cost: convergence behavior of cost function in ILRMA (nIter+1 x 1)
%
% [scalars]
%              I: number of frequency bins, 
%              J: number of time frames
%              M: number of channels (microphones)
%              N: number of sources (equals to M)
%              L: number of bases in NMF model for each source
%
% [matrices]
%              X: observed multichannel spectrograms (I x J x M)
%             pX: permuted observed multichannel spectrograms (M x J x I)
%             pY: permuted separated multisource spectrograms (N x J x I)
%              W: frequency-wise demixing matrix (N x M x I)
%              Y: estimated multisource spectrograms (I x J x N)
%              P: estimated multisource power spectrograms (I x J x N)
%              T: sourcewise basis matrix in NMF (I x L x N)
%              V: sourcewise activation matrix in NMF (L x J x N)
%              R: sourcewise low-rank model spectrogram constructed by T and V (I x J x N)
%              E: identity matrix (N x N)
%              U: model-spectrogram-weighted sample covariance matrix of the mixture (M x M)
%

% Initialization
[I,J,M] = size(X); % I:frequency bins, J: time frames, M: channels
pX = permute(X, [3,2,1]); % permuted X whose dimensions are M x J x I
N = M; % N: number of sources, which equals to M in ILRMA
W = zeros(N,M,I); % frequency-wise demixing matrix
Y = zeros(I,J,N); % estimated spectrograms of sources (Y(i,:,n) =  W(n,:,i)*pX(:,:,i))
for i = 1:I
    W(:,:,i) = eye(N); % initial demixing matrices are set to identity matrices
    Y(i,:,:) = (W(:,:,i)*pX(:,:,i)).'; % initial estimated spectrograms
end
P = max(abs(Y).^2, eps); % power spectrogram of Y
T = max(rand( I, L, N ), eps); % sourcewise basis matrix in NMF
V = max(rand( L, J, N ), eps); % sourcewise activation matrix in NMF
R = zeros(I,J,N); % sourcewise low-rank model spectrogram constructed by T and V (R(:,:,n) = T(:,:,n)*V(:,:,n))
for n = 1:N
    R(:,:,n) = T(:,:,n)*V(:,:,n); % initial source model defined by T and V
end
v = zeros(N,1); % v vector for iterative source steering
cost = zeros(nIter+1, 1);

% Calculate initial cost function value
if drawConv
    cost(1,1) = local_calcCostFunction( P, R, W, I, J );
end

% Optimize parameters in ILRMA (W, T, and V)
fprintf('Iteration:    ');
for iIter = 1:nIter
    fprintf('\b\b\b\b%4d', iIter);
    
    %%%%% Update parameters %%%%%
    for n = 1:N
        %%%%% Update rule of T %%%%%
        T(:,:,n) = T(:,:,n) .* sqrt((P(:,:,n)./(R(:,:,n).^2))*V(:,:,n).' ./ ( (1./R(:,:,n))*V(:,:,n).' ));
        T(:,:,n) = max(T(:,:,n), eps);
        R(:,:,n) = T(:,:,n)*V(:,:,n);
        %%%%% Update rule of V %%%%%
        V(:,:,n) = V(:,:,n) .* sqrt(T(:,:,n).'*(P(:,:,n)./(R(:,:,n).^2)) ./ ( T(:,:,n).'*(1./R(:,:,n)) ));
        V(:,:,n) = max(V(:,:,n), eps);
        R(:,:,n) = T(:,:,n)*V(:,:,n);
        %%%%% Update rule of Y %%%%%
        YY = Y .* conj(Y(:,:,n)); % I x J x N, using implicit expansion (IxJxN .* IxJx1)
        for i = 1:I
            for nn = 1:N % calculate v vector BEGIN
                d = sum((1./(R(i,:,nn))).*real(YY(i,:,n)), 2) / J; % scalar
                if nn ~= n
                    u = sum((1./(R(i,:,nn))).*YY(i,:,nn), 2) / J; % scalar
                    v(nn,1) = u / d;
                else
                    v(nn,1) = 1 - 1/sqrt(d);
                end
            end % calculate v vector END
            Y(i,:,:) = Y(i,:,:) - permute(v.*Y(i,:,n), [3,2,1]); % update Y, usnig implicit expansion (Nx1 .* 1xJ = NxJ -> 1xJxN)
        end
    end
    P = max(abs(Y).^2, eps); % power spectrogram of Y
    
    %%%%% Normalization %%%%%
    if applyNormalize == 1 % average-power-based normalization
        lambda = sqrt(sum(sum(P,1),2)/(I*J)); % 1 x 1 x N
        Y = Y./lambda; % I x J x N (use implicit expansion)
        lambdaPow = lambda.^2; % 1 x 1 x N
        P = P./lambdaPow; % I x J x N (use implicit expansion)
        R = R./lambdaPow; % I x J x N (use implicit expansion)
        T = T./lambdaPow; % I x L x N (use implicit expansion)
    elseif applyNormalize == 2 % back projection
        lambda = local_backProjection(Y, refMixSpecgram, I, N); % N x 1 x I
        pLambda = permute(lambda, [3,2,1]); % I x 1 x N
        Y = Y.*pLambda; % I x J x N (use implicit expansion)
        lambdaPow = abs(pLambda).^2; % I x 1 x N
        P = P.*lambdaPow; % I x J x N (use implicit expansion)
        R = R.*lambdaPow; % I x J x N (use implicit expansion)
        T = T.*lambdaPow; % I x L x N (use implicit expansion)
    end
    
    %%%%% Calculate cost function value %%%%%
    if drawConv
        pY = permute(Y, [3,2,1]); % IxJxN -> NxJxI
        for i = 1:I
            W(:,:,i) = pY(:,:,i)*pX(:,:,i)' / (pX(:,:,i)*pX(:,:,i)'); % derived by "Y(:,:,i) = W(:,:,i)*X(:,:,i)"
        end
        cost(iIter+1,1) = local_calcCostFunction( P, R, W, I, J );
    end
end

% Draw convergence behavior
if drawConv
    figure; plot((0:nIter), cost);
    set(gca, 'FontName', 'Times', 'FontSize', 16);
    xlabel('Number of iterations', 'FontName', 'Arial', 'FontSize', 16);
    ylabel('Value of cost function', 'FontName', 'Arial', 'FontSize', 16);
end
fprintf(' ILRMA1-ISS done.\n');
end

%% Local function for ILRMA type2 (with pertitioning function) based on ISS
function [Y, cost] = local_ILRMA2ISS(X, nIter, K, applyNormalize, drawConv)
% [inputs]
%              X: observed multichannel spectrograms (I x J x M)
%          nIter: number of iterations of the parameter updates
%              K: number of bases in NMF model shared for all the source
% applyNormalize: normalize parameters in each iteration to avoid numerical divergence (0: do not apply, 1: average-power-based normalization, normalization may collapse monotonic decrease of the cost function)
%       drawConv: plot cost function values in each iteration or not (true or false)
%
% [outputs]
%              Y: estimated spectrograms of sources (I x J x N)
%           cost: convergence behavior of cost function in ILRMA (nIter+1 x 1)
%
% [scalars]
%              I: number of frequency bins, 
%              J: number of time frames
%              M: number of channels (microphones)
%              N: number of sources (equals to M)
%              K: number of bases in NMF model shared for all the source
%
% [matrices]
%              X: observed multichannel spectrograms (I x J x M)
%             pX: permuted observed multichannel spectrograms (M x J x I)
%             pY: permuted separated multisource spectrograms (N x J x I)
%              W: frequency-wise demixing matrix (N x M x I)
%              Y: estimated multisource spectrograms (I x J x N)
%              P: estimated multisource power spectrograms (I x J x N)
%              T: basis matrix in NMF (I x K)
%              V: activation matrix in NMF (K x J)
%              Z: partitioning function matrix in NMF that clusters K bases into N sources (N x K)
%              R: sourcewise low-rank model spectrogram constructed by T and V (I x J x N)
%              E: identity matrix (N x N)
%              U: model-spectrogram-weighted sample covariance matrix of the mixture (M x M)
%

% Initialization
[I,J,M] = size(X); % I:frequency bins, J: time frames, M: channels
pX = permute(X, [3,2,1]); % permuted X whose dimensions are M x J x I
N = M; % N: number of sources, which equals to M in ILRMA
W = zeros(N,M,I); % frequency-wise demixing matrix
Y = zeros(I,J,N); % estimated spectrograms of sources (Y(i,:,n) =  W(n,:,i)*pX(:,:,i))
for i = 1:I
    W(:,:,i) = eye(N); % initial demixing matrices are set to identity matrices
    Y(i,:,:) = (W(:,:,i)*pX(:,:,i)).'; % initial estimated spectrograms are set to the observed mixture spectrograms
end
P = max(abs(Y).^2, eps); % power spectrogram of Y
T = max(rand( I, K ), eps); % basis matrix in NMF shared for all the sources
V = max(rand( K, J ), eps); % activation matrix in NMF shared for all the sources
Z = max(rand( N, K ), eps); % partitioning function matrix in NMF that clusters K bases into N sources
Z  = Z./sum(Z,1); % ensure sum_n z_{nk} = 1 (use implicit expansion)
tmpT = zeros(size(T)); % temporal variable used in the update rule of T
tmpV = zeros(size(V)); % temporal variable used in the update rule of V
tmpZ = zeros(size(Z)); % temporal variable used in the update rule of Z 
R = zeros(I,J,N); % sourcewise low-rank model spectrogram constructed by T, V, and Z (R(:,:,n) = T(:,:,n)*V(:,:,n))
for n = 1:N
    R(:,:,n) = (Z(n,:).*T)*V; % initial source model defined by T, V, and Z (use implicit expansion)
end
v = zeros(N,1); % v vector for iterative source steering
cost = zeros(nIter+1, 1);

% Calculate initial cost function value
if drawConv
    cost(1,1) = local_calcCostFunction( P, R, W, I, J );
end

% Optimize parameters in ILRMA (W, T, and V)
fprintf('Iteration:    ');
for iIter = 1:nIter
    fprintf('\b\b\b\b%4d', iIter);
    
    %%%%% Update parameters %%%%%
    %%%%% Update rule of Z %%%%%
    for n = 1:N
        tmpZ(n,:) = (( sum((T.'*(P(:,:,n)./(R(:,:,n).^2))).*V, 2) )./( sum((T.'*(1./R(:,:,n))).*V, 2) )).';
    end
    Z = Z .* sqrt(tmpZ);
    Z  = max(Z./sum(Z,1), eps); % ensure sum_n z_{nk} = 1 (use implicit expansion)
    for n = 1:N
        R(:,:,n) = (Z(n,:).*T)*V; % initial source model defined by NMF (use implicit expansion)
    end
    %%%%% Update rule of T %%%%%
    for i = 1:I
        Pi = squeeze(P(i,:,:)); % J x N
        Ri = squeeze(R(i,:,:)); % J x N
        tmpT(i,:) = (( sum((V*(Pi./(Ri.^2))).*(Z.'), 2) )./( sum((V*(1./Ri)).*(Z.'), 2) )).';
    end
    T = max(T.*sqrt(tmpT), eps);
    for n = 1:N
        R(:,:,n) = (Z(n,:).*T)*V; % initial source model defined by NMF (use implicit expansion)
    end
    %%%%% Update rule of V %%%%%
    for j = 1:J
        Pj = squeeze(P(:,j,:)); % I x N
        Rj = squeeze(R(:,j,:)); % I x N
        tmpV(:,j) = ( sum((T.'*(Pj./(Rj.^2))).*(Z.'), 2) )./( sum((T.'*(1./Rj)).*(Z.'), 2) );
    end
    V = max(V.*sqrt(tmpV), eps);
    for n = 1:N
        R(:,:,n) = (Z(n,:).*T)*V; % initial source model defined by NMF (use implicit expansion)
    end
    %%%%% Update rule of Y %%%%%
    for n=1:N
        YY = Y .* conj(Y(:,:,n)); % I x J x N, using implicit expansion (IxJxN .* IxJx1)
        for i = 1:I
            for nn = 1:N % calculate v vector BEGIN
                d = sum((1./(R(i,:,nn))).*real(YY(i,:,n)), 2) / J; % scalar
                if nn ~= n
                    u = sum((1./(R(i,:,nn))).*YY(i,:,nn), 2) / J; % scalar
                    v(nn,1) = u / d;
                else
                    v(nn,1) = 1 - 1/sqrt(d);
                end
            end % calculate v vector END
            Y(i,:,:) = Y(i,:,:) - permute(v.*Y(i,:,n), [3,2,1]); % update Y, usnig implicit expansion (Nx1 .* 1xJ = NxJ -> 1xJxN)
        end
    end
    P = max(abs(Y).^2, eps); % power spectrogram of Y
    
    %%%%% Normalization %%%%%
    if applyNormalize == 1
        lambda = sqrt(sum(sum(P,1),2)/(I*J)); % 1 x 1 x N
        Y = Y./lambda; % I x J x N (use implicit expansion)
        P = P./lambda.^2; % I x J x N (use implicit expansion)
        R = R./lambda.^2; % I x J x N (use implicit expansion)
        Zlambda = Z./(squeeze(lambda).^2); % N x K
        ZlambdaSum = sum(Zlambda,1); % 1 x K
        T = T.*ZlambdaSum; % I x K (use implicit expansion)
        Z = Zlambda./ZlambdaSum; % N x K (use implicit expansion)
    end
    
    %%%%% Calculate cost function value %%%%%
    if drawConv
        pY = permute(Y, [3,2,1]); % IxJxN -> NxJxI
        for i = 1:I
            W(:,:,i) = pY(:,:,i)*pX(:,:,i)' / (pX(:,:,i)*pX(:,:,i)'); % derived by "Y(:,:,i) = W(:,:,i)*X(:,:,i)"
        end
        cost(iIter+1,1) = local_calcCostFunction( P, R, W, I, J );
    end
end

% Draw convergence behavior
if drawConv
    figure; plot((0:nIter), cost);
    set(gca, 'FontName', 'Times', 'FontSize', 16);
    xlabel('Number of iterations', 'FontName', 'Arial', 'FontSize', 16);
    ylabel('Value of cost function', 'FontName', 'Arial', 'FontSize', 16);
end
fprintf(' ILRMA2-ISS done.\n');
end

%% Local function for calculating cost function value in ILRMA
function [ cost ] = local_calcCostFunction(P, R, W, I, J)
logDetAbsW = zeros(I,1);
for i = 1:I
    logDetAbsW(i,1) = log(max(abs(det(W(:,:,i))), eps));
end
cost = sum(sum(sum(P./R+log(R),3),2),1) - 2*J*sum(logDetAbsW, 1);
end

%% Local function for applying back projection and returns frequency-wise coefficients
function [ D ] = local_backProjection(Y, X, I, N)
D = zeros(I,N);
for i = 1:I
    Yi = squeeze(Y(i,:,:)).'; % N x J
    D(i,:) = X(i,:,1)*Yi'/(Yi*Yi'); % 1 x N
end
D(isnan(D) | isinf(D)) = 0; % replace NaN and Inf to 0
D = permute(D, [2,3,1]); % N x 1 x I
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%