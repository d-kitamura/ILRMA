%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample program for blind source separation using independent low-rank   %
% matrix analysis (ILRMA)                                                 %
%                                                                         %
% Coded by D. Kitamura (d-kitamura@ieee.org) on 1 Apr, 2018 (ver1.0).     %
%                                                                         %
% Copyright 2018 Daichi Kitamura                                          %
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
% http://d-kitamura.net/en/demo_rank1_en.htm                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;

% Set parameters
seed = 1; % pseudo random seed
refMic = 1; % reference microphone for back projection
fsResample = 16000; % resampling frequency [Hz]
ns = 2; % number of sources
fftSize = 4096; % window length in STFT [points]
shiftSize = 2048; % shift length in STFT [points]
nb = 10; % number of bases (for type=1, nb is # of bases for "each" source. for type=2, nb is # of bases for "all" sources)
it = 100; % number of iterations (define by checking convergence behavior with drawConv=true)
type = 1; % 1 or 2 (1: ILRMA w/o partitioning function, 2: ILRMA with partitioning function)
drawConv = true; % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)
normalize = true; % true or false (true: apply normalization in each iteration of ILRMA to improve numerical stability, but the monotonic decrease of the cost function may be lost. false: do not apply normalization)

% Fix random seed
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed))

% Input data and resample
[sig(:,:,1), fs] = audioread('./input/drums.wav'); % signal x channel x source (source image)
[sig(:,:,2), fs] = audioread('./input/piano.wav'); % signal x channel x source (source image)
sig_resample(:,:,1) = resample(sig(:,:,1), fsResample, fs, 100); % resampling for reducing computational cost
sig_resample(:,:,2) = resample(sig(:,:,2), fsResample, fs, 100); % resampling for reducing computational cost

% Mixing source images in each channel to produce observed signal
mix(:,1) = sig_resample(:,1,1) + sig_resample(:,1,2);
mix(:,2) = sig_resample(:,2,1) + sig_resample(:,2,2);
if abs(max(max(mix))) > 1 % check clipping
    error('Cliping detected while mixing.\n');
end

% Blind source separation based on ILRMA
[sep, cost] = bss_ILRMA(mix,ns,nb,fftSize,shiftSize,it,type,refMic,drawConv,normalize);
% [sep, cost] = bss_ILRMA_readable(mix,ns,nb,fftSize,shiftSize,it,type,refMic,drawConv,normalize);

% Output separated signals
outputDir = sprintf('./output');
if ~isdir( outputDir )
    mkdir( outputDir );
end
audiowrite(sprintf('%s/observedMixture.wav', outputDir), mix, fsResample); % observed signal
audiowrite(sprintf('%s/originalSource1.wav', outputDir), sig_resample(:,refMic,1), fsResample); % source signal 1
audiowrite(sprintf('%s/originalSource2.wav', outputDir), sig_resample(:,refMic,2), fsResample); % source signal 2
audiowrite(sprintf('%s/estimatedSignal1.wav', outputDir), sep(:,1), fsResample); % estimated signal 1
audiowrite(sprintf('%s/estimatedSignal2.wav', outputDir), sep(:,2), fsResample); % estimated signal 2

fprintf('The files are saved in "./output".\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%