function waveform = ISTFT(S, shiftSize, window, length)
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
%   waveform = ISTFT(S)
%   waveform = ISTFT(S, shiftSize)
%   waveform = ISTFT(S, shiftSize, window)
%   waveform = ISTFT(S, shiftSize, window, length)
%
% [inputs]
%           S: STFT of input signal (frequency bin (fftSize/2+1) x time frame x channel)
%   shiftSize: frame shift length (default: fftSize/2)
%      window: window function used in STFT (fftSize x 1) or choose used function from below:
%              'hamming'    : Hamming window (default)
%              'hann'       : von Hann window
%              'rectangular': rectangular window
%              'blackman'   : Blackman window
%              'sine'       : sine window
%      length: length of original signal (before STFT) (default: the same as that of output signal)
%
% [outputs]
%   waveform: time-domain waveform of the input spectrogram (signal x channel)
%

% Check errors and set default values
if (nargin < 1)
    error('Too few input arguments.\n');
end
[freq, frames, nch] = size(S);
if (nch > freq)
    error('The input spectrogram might be wrong. The size of it must be (freq x frame x ch).\n');
end
if (isreal(S) == 1)
    error('The input spectrogram might be wrong. It does not complex-valued matrix.\n');
end
if (mod(freq,2) == 0)
    error('The number of rows of the first argument must be an odd number because it is (frame length /2)+1.\n');
end
fftSize = (freq-1) * 2;
if (nargin < 2)
    shiftSize = fftSize/2;
elseif (mod(fftSize,shiftSize) ~= 0)
    error('The frame length must be dividable by the second argument.\n');
end
if (nargin<3)
    window = hamming_local(fftSize); % default window
else
    if (isnumeric(window))
        if (size(window, 1) ~= fftSize)
            error('The length of the third argument must be the same as FFT size used in STFT.\n');
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
                error('Unsupported window is required. Type "help STFT" and check options.\n')
        end
    end
end
invWindow = optSynWnd_local( window, shiftSize );

% Inverse STFT
spectrum = zeros(fftSize,1);
tmpSignal = zeros((frames-1)*shiftSize+fftSize,nch);
for ch = 1:nch
    for i = 1:frames
        spectrum(1:fftSize/2+1,1) = S(:,i,ch);
        spectrum(1,1) = spectrum(1,1)/2;
        spectrum(fftSize/2+1,1) = spectrum(fftSize/2+1,1)/2;
        sp = (i-1)*shiftSize;
        tmpSignal(sp+1:sp+fftSize,ch) = tmpSignal(sp+1:sp+fftSize,ch) + real(ifft(spectrum,fftSize).*invWindow)*2;
    end
end
waveform = tmpSignal(fftSize-shiftSize+1:(frames-1)*shiftSize+fftSize,:); % discard padded zeros in STFT

% Discarding padded zeros in the end of the signal
if (nargin==4)
    waveform = waveform(1:length,:);
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

function synthesizedWindow = optSynWnd_local(analysisWindow,shiftSize) % based on minimum distortion
fftSize = size(analysisWindow,1);
synthesizedWindow = zeros(fftSize,1);
for i = 1:shiftSize
    amp = 0.0;
    for j = 1:fftSize/shiftSize
        amp = amp + analysisWindow(i+(j-1)*shiftSize,1)*analysisWindow(i+(j-1)*shiftSize,1);
    end
    for j = 1:fftSize/shiftSize
        synthesizedWindow(i+(j-1)*shiftSize,1) = analysisWindow(i+(j-1)*shiftSize,1)/amp;
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%