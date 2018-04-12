%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Blind audio source separation based on independent low-rank matrix      %
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

Contents:
	- input [dir]:		includes test audio signals (reverberation time is around 300 ms)
	- reference [dir]:	includes reference documents for ILRMA
	- bss_ILRMA.m:		apply pre- and post-processing for blind source separation (STFT, whitening, ILRMA, back projection, and ISTFT)
	- ILRMA.m:			function of ILRMA (fast and heuristic implementation)
	- ILRMA_readable.m:	function of ILRMA (slow but somewhat readable implementation)
	- ISTFT.m:			inverse short-time Fourier transform
	- main.m:			main script with parameter settings
	- backProjection.m:	back projection technique (fixing frequency-wise scales)
	- STFT.m:			short-time Fourier transform
	- whitening.m:		applying principal component analysis for decorrelating observed multichannel signal

Note:
A parameter "normalize" in 47th line of main.m is very important. 
This parameter controls whether apply normalization process (lines from 187th to 192nd in ILRMA.m) in each iteration of ILRMA or not.
Normalization process will improve the numerical stability of the algorithm, but the monotonic decrease of the cost function in the update algorithm may be lost. 
Useful information can be found at:
http://d-kitamura.net/pdf/misc/AlgorithmsForIndependentLowRankMatrixAnalysis.pdf

Daichi Kitamura
d-kitamura@ieee.org