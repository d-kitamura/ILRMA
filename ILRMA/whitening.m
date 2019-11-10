function Y = whitening(X, dnum)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Whitening pre-process based on principal component analysis (PCA)       %
% This function reduces a dimension of multichannel input signal by       %
% applying PCA so that the number of channels is equal to the number of   %
% sources for determined blind source separation.                         %
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
%   Y = whitening(X, dnum)
%
% [inputs]
%      X: input multichannel spectrogram (frequency bin x time frame x channel)
%   dnum: number of dimensions to which X is projected (scalar)
%
% [outputs]
%      Y: output spectrogram (frequency bin x time frame x dnum)
%

[I,J,M] = size(X);
Y = zeros(I,J,dnum);
for i=1:I
    Xi = squeeze(X(i,:,:)).'; % M x J
    V = Xi*(Xi')/J; % covariance matrix
    [P,D] = eig(V);
    
    % sort eigenvalues in ascending order
    eig_val = diag(D);
    [eig_val,idx] = sort(eig_val);
    D = D(idx,idx);
    P = P(:,idx);
    D2 = D(M-dnum+1:M,M-dnum+1:M);
    P2 = P(:,M-dnum+1:M);
    
    Y(i,:,:) = (sqrt(D2)\(P2')*Xi).';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%