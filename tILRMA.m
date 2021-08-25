function [estSig, cost] = tILRMA(mixSig, nSrc, sampFreq, nBases, dofParam, sigDom, fftSize, shiftSize, windowType, nIter, refMic, applyNormalize, applyWhitening, drawConv)
% Blind source separation with independent low-rank matrix analysis (ILRMA)
% based on Student's t distribution
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
% # Original paper
% D. Kitamura, S. Mogami, Y. Mitsui, N. Takamune, H. Saruwatari, N. Ono, 
% Y. Takahashi, and K. Kondo, "Generalized independent low-rank matrix 
% analysis using heavy-tailed distributions for blind source separation," 
% EURASIP Journal on Advances in Signal Processing, vol. 2018, no. 28, 
% p. 25, May 2018.
%
% See also:
% http://d-kitamura.net
%
% [syntax]
%   [estSig,cost] = tILRMA(mixSig,nSrc,sampFreq,nBases,fftSize,shiftSize,windowType,nIter,ilrmaType,refMic,applyNormalize,applyWhitening,drawConv)
%
% [inputs]
%         mixSig: observed mixture (sigLen x nCh)
%           nSrc: number of sources in the mixture (scalar)
%       sampFreq: sampling frequency [Hz] of mixSig (scalar)
%         nBases: number of bases in NMF model (scalar, # of bases for "each" source when ilrmaType=1, and # of bases for "all" sources when ilrmaType=2, default: 4)
%       dofParam: degree-of-freedom parameter that defines shape of Student's t distribution (scalar, default: 1)
%         sigDom: domain of signal for source model (scalar, default: 2)
%        fftSize: window length [points] in STFT (scalar, default: next higher power of 2 that exceeds 0.256*sampFreq)
%      shiftSize: shift length [points] in STFT (scalar, default: fftSize/2)
%     windowType: window function used in STFT (name of window function, default: 'hamming')
%          nIter: number of iterations in the parameter update in ILRMA (scalar, default: 100)
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
    dofParam (1,1) double = 1
    sigDom (1,1) double  = 2
    fftSize (1,1) double {mustBeInteger(fftSize)} = 2^nextpow2(0.256*sampFreq)
    shiftSize (1,1) double {mustBeInteger(shiftSize)} = fftSize/2
    windowType char {mustBeMember(windowType,{'hamming','hann','rectangular','blackman','sine'})} = 'hamming'
    nIter (1,1) double {mustBeInteger(nIter)} = 100
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
if dofParam <= 0; error("The dgree-of-freedom parameter (dofParam) must be a positive value.\n"); end
if sigDom <= 0; error("The domain of signal for source model (sigDom) must be a positive value.\n"); end
if fftSize < 1; error("The FFT length in STFT (fftSize) must be a positive integer value.\n"); end
if shiftSize < 1; error("The shift length in STFT (shiftSize) must be a positive integer value.\n"); end
if nIter < 1; error("The number of iterations (nIter) must be a positive integer value.\n"); end
if refMic < 1 || refMic > nCh; error("The reference microphone must be an integer between 1 and nCh.\n"); end
if applyNormalize ~= 0 && applyNormalize ~= 1 && applyNormalize ~= 2; error("The normalization type (applyNormalize) must be set to 0, 1, or 2.\n"); end

% Apply multichannel short-time Fourier transform (STFT)
[mixSpecgram, windowInStft] = STFT(mixSig, fftSize, shiftSize, windowType);

% Apply whitening (decorrelate X so that the correlation matrix becomes an identity matrix) based on principal component analysis
if applyWhitening
    inputMixSpecgram = whitening(mixSpecgram, nSrc); % apply whitening, where dimension is reduced from nCh to nSrc when nSrc < nCh
else
    inputMixSpecgram = mixSpecgram(:,:,1:nSrc); % when nSrc < nCh, only mixSpecgram(:,:,1:nSrc) is input to ILRMA so that the number of microphones equals to the number of sources (determined condition)
end

% Apply t-ILRMA
[estSpecgram, cost] = local_tILRMA(inputMixSpecgram, nIter, nBases, dofParam, sigDom, applyNormalize, drawConv, mixSpecgram(:,:,refMic));

% Apply back projection (fix the scale ambiguity using the reference microphone channel)
scaleFixedSepSpecgram = backProjection(estSpecgram, mixSpecgram(:,:,refMic)); % scale-fixed estimated signal

% Inverse STFT for each source
estSig = ISTFT(scaleFixedSepSpecgram, shiftSize, windowInStft, sigLen);
end

%% Local function for t-ILRMA (without pertitioning function)
function [Y, cost] = local_tILRMA(X, nIter, L, nu, p, applyNormalize, drawConv, refMixSpecgram)
% [inputs]
%              X: observed multichannel spectrograms (I x J x M)
%          nIter: number of iterations of the parameter updates
%              L: number of bases in NMF model for each source
%             nu: degree-of-freedom parameter that defines shape of Student's t distribution
%              p: domain of signal for source model
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
%             nu: degree-of-freedom parameter (1: Cauchy, inf: Gauss)
%              p: signal domain (2: power, 1: amplitude)
%
% [matrices]
%              X: observed multichannel spectrograms (I x J x M)
%             pX: permuted observed multichannel spectrograms (M x J x I)
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
B = zeros(I,J,N); % sourcewise temporary model
for n = 1:N
    R(:,:,n) = T(:,:,n)*V(:,:,n); % initial source model defined by T and V
    B(:,:,n) = (nu/(nu+2))*R(:,:,n).^(2/p) + 2/(nu+2)*P(:,:,n);
end
E = eye(N); % identity matrix for e_n
cost = zeros(nIter+1, 1);

% Calculate initial cost function value
if drawConv
    cost(1,1) = local_calcCostFunction( P, R, W, I, J, nu, p );
end

% Optimize parameters in ILRMA (W, T, and V)
fprintf('Iteration:    ');
for iIter = 1:nIter
    fprintf('\b\b\b\b%4d', iIter);
    
    %%%%% Update parameters %%%%%
    for n = 1:N
        %%%%% Update rule of T %%%%%
        T(:,:,n) = T(:,:,n) .* ((P(:,:,n)./(B(:,:,n).*R(:,:,n)))*V(:,:,n).' ./ ( (1./R(:,:,n))*V(:,:,n).' )).^(p/(p+2));
        T(:,:,n) = max(T(:,:,n), eps);
        R(:,:,n) = T(:,:,n)*V(:,:,n);
        B(:,:,n) = (nu/(nu+2))*R(:,:,n).^(2/p) + 2/(nu+2)*P(:,:,n);
        %%%%% Update rule of V %%%%%
        V(:,:,n) = V(:,:,n) .* (T(:,:,n).'*(P(:,:,n)./(B(:,:,n).*R(:,:,n))) ./ ( T(:,:,n).'*(1./R(:,:,n)) )).^(p/(p+2));
        V(:,:,n) = max(V(:,:,n), eps);
        R(:,:,n) = T(:,:,n)*V(:,:,n);
        B(:,:,n) = (nu/(nu+2))*R(:,:,n).^(2/p) + 2/(nu+2)*P(:,:,n);
        %%%%% Update rule of W %%%%%
        for i = 1:I
            zeta = 1+(2/nu)*P(i,:,n)./R(i,:,n).^(2/p); % zeta: 1 x J vector
            U = (1/J)*(2/nu+1)*(pX(:,:,i).*(1./(zeta.*R(i,:,n).^(2/p))))*pX(:,:,i)'; % U: M x M matrix (use implicit expansion)
            w = (W(:,:,i)*U)\E(:,n); % w: M x 1 vector
            w = w/sqrt(w'*U*w); % w: M x 1 vector
            W(n,:,i) = w'; % w': 1 x M vector
        end
    end
    for i = 1:I
        Y(i,:,:) = (W(:,:,i)*pX(:,:,i)).'; % temporal estimated spectrograms of sources
    end
    P = max(abs(Y).^2, eps); % power spectrogram of Y
    
    %%%%% Normalization %%%%%
    if applyNormalize == 1 % average-power-based normalization
        lambda = sqrt(sum(sum(P,1),2)/(I*J)); % 1 x 1 x N
        W = W./squeeze(lambda); % N x M x I (use implicit expansion)
        lambdaPow = lambda.^p; % 1 x 1 x N
        P = P./lambdaPow; % I x J x N (use implicit expansion)
        R = R./lambdaPow; % I x J x N (use implicit expansion)
        T = T./lambdaPow; % I x L x N (use implicit expansion)
    elseif applyNormalize == 2 % back projection
        lambda = local_backProjection(Y, refMixSpecgram, I, N); % N x 1 x I
        W = W.*lambda; % N x M x I (use implicit expansion)
        lambdaPow = permute(abs(lambda).^p, [3,2,1]); % I x 1 x N
        P = P.*lambdaPow; % I x J x N (use implicit expansion)
        R = R.*lambdaPow; % I x J x N (use implicit expansion)
        T = T.*lambdaPow; % I x L x N (use implicit expansion)
    end
    
    %%%%% Calculate cost function value %%%%%
    if drawConv
        cost(iIter+1,1) = local_calcCostFunction( P, R, W, I, J, nu, p );
    end
end

% Draw convergence behavior
if drawConv
    figure; plot((0:nIter), cost);
    set(gca, 'FontName', 'Times', 'FontSize', 16);
    xlabel('Number of iterations', 'FontName', 'Arial', 'FontSize', 16);
    ylabel('Value of cost function', 'FontName', 'Arial', 'FontSize', 16);
end
fprintf(' t-ILRMA done.\n');
end

%% Local function for calculating cost function value in ILRMA
function [ cost ] = local_calcCostFunction(P, R, W, I, J, nu, p)
logDetAbsW = zeros(I,1);
for i = 1:I
    logDetAbsW(i,1) = log(max(abs(det(W(:,:,i))), eps));
end
cost = sum(sum(sum((1+nu/2)*log( 1+(2/nu)*P./(R.^(2/p)) ) + (2/p)*log(R),3),2),1) - 2*J*sum(logDetAbsW, 1);
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