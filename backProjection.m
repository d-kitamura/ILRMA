function [Z] = backProjection(Y, X)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Back projection technique for fixing the frequency-wise scales of the   %
% signals estimated by ICA-based blind source separation techniques.      %
% Both monaural and multichannel outputs are supported.                   %
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
%   [Z, D] = backProjection(Y, X)
%
% [inputs]
%   Y: estimated (separated) signals (frequency bin x time frame x source)
%   X: reference channel of observed (mixture) signal (frequency bin x time frame x 1)
%      or observed multichannel signals (frequency bin x time frame x channels)
%
% [outputs]
%   Z: scale-fitted estimated signals (frequency bin x time frame x source)
%      or scale-fitted estimated source images (frequency bin x time frame x source x channel)
%

% check errors
if (nargin<2)
    error('Too few input arguments.\n');
end
[I,J,M] = size(Y); % frequency bin x time frame x source

% back projection
if (size(X,3)==1) % calculate scale-fixed estimated signals using X(:,:,1)
    A = zeros(1,M,I);
    Z = zeros(I,J,M);
    for i=1:I
        Yi = squeeze(Y(i,:,:)).'; % channels x frames (M x J)
        A(1,:,i) = X(i,:,1)*Yi'/(Yi*Yi');
    end
    A(isnan(A)) = 0;
    A(isinf(A)) = 0;
    for m=1:M
        for i=1:I
            Z(i,:,m) = A(1,m,i)*Y(i,:,m);
        end
    end
elseif (size(X,3)==M) % calculate scale-fixed source images of estimated signals
    A = zeros(M,M,I);
    Z = zeros(I,J,M,M); % frequency bin x time frame x source x channel
    for i=1:I
        for m=1:M
            Yi = squeeze(Y(i,:,:)).'; % channels x frames (M x J)
            A(m,:,i) = X(i,:,m)*Yi'/(Yi*Yi');
        end
    end
    A(isnan(A)) = 0;
    A(isinf(A)) = 0;
    for n=1:M
        for m=1:M
            for i=1:I
                Z(i,:,n,m) = A(m,n,i)*Y(i,:,n);
            end
        end
    end
else
    error('The number of channels in X must be 1 or equal to that in Y.\n');
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%