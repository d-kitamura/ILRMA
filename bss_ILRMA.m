function [sep, cost] = bss_ILRMA(mix, ns, nb, fftSize, shiftSize, it, type, refMic, drawConv, normalize)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Blind source separation using independent low-rank matrix analysis      %
% (ILRMA)                                                                 %
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
%
% [syntax]
%   [sep, cost] = bss_ILRMA(mix, ns, nb, fftSize, shiftSize, it, type, draw, stable)
%
% [inputs]
%        mix: observed mixture (signal x channel)
%         ns: number of sources (scalar)
%         nb: number of bases (scalar, # of bases for "each" source when type=1, and # of bases for "all" sources when type=2)
%    fftSize: window length in STFT (scalar)
%  shiftSize: shift length in STFT (scalar)
%         it: number of iterations (scalar)
%       type: without or with partitioning function (1: ILRMA without partitioning function (ILRMA1), 2: ILRMA with partitioning function (ILRMA2))
%     refMic: reference microphone for applying back projection
%   drawConv: plot cost function values in each iteration or not (true or false)
%  normalize: normalize variables in each iteration to avoid numerical divergence or not (true or false, normalization may collapse monotonic decrease of the cost function)
%
% [outputs]
%        sep: estimated signals (signal x channel x ns)
%       cost: convergence behavior of cost function in ILRMA (it+1 x 1)
%

% Short-time Fourier transform
[X, window] = STFT( mix, fftSize, shiftSize, 'hamming' );

% Whitening
Xwhite = whitening( X, ns ); % decorrelate input multichannel signal by applying principal component analysis

% ILRMA
[Y, cost] = ILRMA( Xwhite, type, it, nb, drawConv, normalize );
% [Y, cost] = ILRMA_readable( Xwhite, type, it, nb, drawConv, normalize );

% Back projection (fixing scale ambiguity using reference microphone)
Z = backProjection( Y, X(:,:,refMic) ); % scale-fixed estimated signal

% Inverse STFT for each source
sep = ISTFT( Z, shiftSize, window, size(mix,1) );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%