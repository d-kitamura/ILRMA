function [sep, cost] = bss_consistentILRMA(mix, nb, fftSize, shiftSize, it, refMic, drawConv)
% Blind source separation using independent low-rank matrix analysis 
% (ILRMA)
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% Copyright 2020 Daichi Kitamura
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
%   [sep, cost] = bss_consistentILRMA(mix, nb, fftSize, shiftSize, it, refMic, drawConv)
%
% [inputs]
%        mix: observed mixture (signal x channel)
%         nb: number of bases (scalar, # of bases for "each" source when type=1, and # of bases for "all" sources when type=2)
%    fftSize: window length in STFT (scalar)
%  shiftSize: shift length in STFT (scalar)
%         it: number of iterations (scalar)
%     refMic: reference microphone for applying back projection
%   drawConv: plot cost function values in each iteration or not (true or false)
%
% [outputs]
%        sep: estimated signals (signal x channel x ns)
%       cost: convergence behavior of cost function in ILRMA (it+1 x 1)
%

mixLen = size(mix, 1);

% Short-time Fourier transform
[X, window] = STFT( mix, fftSize, shiftSize, 'hamming' );

% ILRMA with spectrogram consistency
[Y, cost] = consistentILRMA( X, it, nb, fftSize, shiftSize, window, mixLen, refMic, drawConv );

% Inverse STFT for each source
sep = ISTFT( Y, shiftSize, window, size(mix,1) );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%