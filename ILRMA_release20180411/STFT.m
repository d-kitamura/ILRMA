function [S, window] = STFT(signal, fftSize, shiftSize, window)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Short-time Fourier transform (STFT)                                     %
% Both monaural and multichannel signals are supported.                   %
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
%   [S, window] = STFT(signal, fftSize)
%   [S, window] = STFT(signal, fftSize, shiftSize)
%   [S, window] = STFT(signal, fftSize, shiftSize, window)
%
% [inputs]
%     signal: input signal (signal x channel)
%    fftSize: frame length (even number, must be dividable by shiftSize)
%  shiftSize: frame shift (default: fftSize/2)
%     window: arbitrary window function (fftSize x 1) or choose desired function from below:
%             'hamming'    : Hamming window (default)
%             'hann'       : von Hann window
%             'rectangular': rectangular window
%             'blackman'   : Blackman window
%             'sine'       : sine window
%
% [outputs]
%          S: spectrogram of input signal (frequency bin (fftSize/2+1) x time frame x channel)
%     window: window function used in STFT (fftSize x 1)
%

% Check errors and set default values
if (nargin < 2)
    error('Too few input arguments.\n');
end
if (mod(fftSize,2) ~= 0)
    error ('The third argument must be an even number.\n');
end
if (nargin<3)
    shiftSize = fftSize / 2;
elseif (mod(fftSize,shiftSize) ~= 0)
    error('The second argument must be dividable by the third argument.\n');
end
if (nargin<4)
    window = hamming_local(fftSize); % default window
else
    if (isnumeric(window))
        if (size(window, 1) ~= fftSize)
            error('The length of the forth argument must be the same as that of the second argument.\n');
        end
    else
        switch window
            case 'hamming'
                window = hamming_local(fftSize);
            case 'hann'
                window = hann_local(fftSize);
            case 'rectangular'
                window = rectangular_local(fftSize);
            case 'blackman'
                window = blackman_local(fftSize);
            case 'sine'
                window = sine_local(fftSize);
            otherwise
                error('Unsupported window is requested. Type "help STFT" and check options.\n')
        end
    end
end

% Pad zeros
nch = size(signal,2);
zeroPadSize = fftSize - shiftSize;
signal = [zeros(zeroPadSize,nch); signal; zeros(fftSize,nch)];
length = size(signal,1);

% Calculate STFT
frames = floor( (length - fftSize + shiftSize) / shiftSize ); % number of frames
S = zeros( fftSize/2+1, frames, nch ); % memory allocation
for ch = 1:nch
    for j = 1:frames
        sp = (j-1)*shiftSize;
        spectrum = fft( signal(sp+1:sp+fftSize,ch) .* window );
        S(:,j,ch) = spectrum(1:fftSize/2+1, 1); % freq. x frame x ch
    end
end
end

%% Local functions
function window = hamming_local(fftSize)
t = linspace(0,1,fftSize+1).'; % periodic (produce L+1 window and return L window)
window = 0.54*ones(fftSize,1) - 0.46*cos(2.0*pi*t(1:fftSize));
end

function window = hann_local(fftSize)
t = linspace(0,1,fftSize+1).'; % periodic (produce L+1 window and return L window)
window = max(0.5*ones(fftSize,1) - 0.5*cos(2.0*pi*t(1:fftSize)),eps);
end

function window = rectangular_local(fftSize)
window = ones(fftSize,1);
end

function window = blackman_local(fftSize)
t = linspace(0,1,fftSize+1).'; % periodic (produce L+1 window and return L window)
window = max(0.42*ones(fftSize,1) - 0.5*cos(2.0*pi*t(1:fftSize)) + 0.08*cos(4.0*pi*t(1:fftSize)),eps);
end

function window = sine_local(fftSize)
t = linspace(0,1,fftSize+1).'; % periodic (produce L+1 window and return L window)
window = max(sin(pi*t(1:fftSize)),eps);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%