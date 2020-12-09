%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample program for blind source separation using independent low-rank   %
% matrix analysis (ILRMA)                                                 %
%                                                                         %
% Coded by D. Kitamura (d-kitamura@ieee.org)                              %
%                                                                         %
% Copyright 2020 Daichi Kitamura                                          %
%                                                                         %
% These programs are distributed only for academic research at            %
% universities and research institutions.                                 %
% It is not allowed to use or modify these programs for commercial or     %
% industrial purpose without our permission.                              %
% When you use or modify these programs and write research articles,      %
% cite the following references:                                          %
%                                                                         %
% # Original paper (The algorithm was called "Rank-1 MNMF" in this paper) %
% D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined  %
% blind source separation unifying independent vector analysis and        %
% nonnegative matrix factorization," IEEE/ACM Trans. ASLP, vol. 24,       %
% no. 9, pp. 1626-1641, September 2016.                                   %
%                                                                         %
% # Book chapter (The algorithm was renamed as "ILRMA")                   %
% D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined  %
% blind source separation with independent low-rank matrix analysis,"     %
% Audio Source Separation. Signals and Communication Technology.,         %
% S. Makino, Ed. Springer, Cham, pp. 125-155, March 2018.                 %
%                                                                         %
% See also:                                                               %
% http://d-kitamura.net                                                   %
% http://d-kitamura.net/demo-ILRMA_en.html                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;

% Set parameters
seed = 1; % pseudo random seed
refMic = 1; % reference microphone for back projection
resampFreq = 16000; % resampling frequency [Hz]
nSrc = 2; % number of sources
fftSize = 4096; % window length in STFT [points]
shiftSize = 2048; % shift length in STFT [points]
windowType = "hamming"; % window function used in STFT
nBases = 10; % number of bases (for ilrmaType=1, nBases is # of bases for "each" source. for ilrmaType=2, nBases is # of bases for "all" sources)
nIter = 100; % number of iterations (define by checking convergence behavior with drawConv=true)
ilrmaType = 1; % 1 or 2 (1: ILRMA w/o partitioning function, 2: ILRMA with partitioning function)
applyNormalize = 1; % 0 or 1 or 2 (0: do not apply normalization in each iteration, 1: apply average-power-based normalization in each iteration to improve numerical stability (the monotonic decrease of the cost function may be lost), 2: apply back projection in each iteration)
applyWhitening = false; % true or false (true: apply whitening to the observed multichannel spectrograms)
drawConv = true; % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)

% Fix random seed
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed))

% Input data and resample
[srcSig(:,:,1), sampFreq] = audioread('./input/drums.wav'); % signal x channel x source (source image)
[srcSig(:,:,2), sampFreq] = audioread('./input/piano.wav'); % signal x channel x source (source image)
srcSigResample(:,:,1) = resample(srcSig(:,:,1), resampFreq, sampFreq, 100); % resampling for reducing computational cost
srcSigResample(:,:,2) = resample(srcSig(:,:,2), resampFreq, sampFreq, 100); % resampling for reducing computational cost

% Mix source images of each channel to produce observed mixture signal
mixSig(:,1) = srcSigResample(:,1,1) + srcSigResample(:,1,2);
mixSig(:,2) = srcSigResample(:,2,1) + srcSigResample(:,2,2);
if abs(max(max(mixSig))) > 1 % check clipping
    error('Cliping detected while mixing.\n');
end

% Blind source separation based on ILRMA
[estSig, cost] = ILRMA(mixSig, nSrc, resampFreq, nBases, fftSize, shiftSize, windowType, nIter, ilrmaType, refMic, applyNormalize, applyWhitening, drawConv);

% Blind source separation based on consistent ILRMA
% [estSig, cost] = consistentILRMA(mixSig, nSrc, resampFreq, nBases, fftSize, shiftSize, windowType, nIter, refMic, applyWhitening, drawConv);

% Output separated signals
outputDir = sprintf('./output');
if ~isdir( outputDir )
    mkdir( outputDir );
end
audiowrite(sprintf('%s/observedMixture.wav', outputDir), mixSig, resampFreq); % observed signal
audiowrite(sprintf('%s/originalSource1.wav', outputDir), srcSigResample(:,refMic,1), resampFreq); % source signal 1
audiowrite(sprintf('%s/originalSource2.wav', outputDir), srcSigResample(:,refMic,2), resampFreq); % source signal 2
audiowrite(sprintf('%s/estimatedSignal1.wav', outputDir), estSig(:,1), resampFreq); % estimated signal 1
audiowrite(sprintf('%s/estimatedSignal2.wav', outputDir), estSig(:,2), resampFreq); % estimated signal 2

fprintf('The files are saved in "./output".\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%