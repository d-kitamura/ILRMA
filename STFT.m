function [spectrogram,analyWindow] = STFT(signal,fftSize,shiftSize,window)
%
% Short-time Fourier transform
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% See also:
% http://d-kitamura.net
%
% [syntax]
%   [spectrogram,analyWindow] = STFT(signal,fftSize)
%   [spectrogram,analyWindow] = STFT(signal,fftSize,shiftSize)
%   [spectrogram,analyWindow] = STFT(signal,fftSize,shiftSize,window)
%
% [inputs]
%       signal: input signal (length x channels)
%      fftSize: frame length (even number, must be dividable by shiftSize)
%    shiftSize: frame shift (default: fftSize/2)
%       window: arbitrary analysis window function (fftSize x 1) or choose function from below:
%               'hamming'    : Hamming window (default)
%               'hann'       : von Hann window
%               'rectangular': rectangular window
%               'blackman'   : Blackman window
%               'sine'       : sine window
%
% [outputs]
%  spectrogram: spectrogram of input signal (frequency bins (fftSize/2+1) x time frames x channels)
%  analyWindow: analysis window function used in STFT (fftSize x 1) and can be used for calculating optimal synthesis window
%

% Check errors and set default values
if (nargin < 2)
    error('Too few input arguments.\n');
end
if (mod(fftSize,2) ~= 0)
    error ('fftSize must be an even number.\n');
end
if (nargin < 3)
    shiftSize = fftSize / 2;
elseif (mod(fftSize,shiftSize) ~= 0)
    error('fftSize must be dividable by shiftSize.\n');
end
if (nargin<4)
    analyWindow = hamming_local(fftSize); % default analysis window
else
    if (isnumeric(window))
        if (size(window, 1) ~= fftSize)
            error('The length of analysis window must be the same as that of fftSize.\n');
        else
            analyWindow = window;
        end
    else
        switch window
            case 'hamming'
                analyWindow = hamming_local(fftSize);
            case 'hann'
                analyWindow = hann_local(fftSize);
            case 'rectangular'
                analyWindow = rectangular_local(fftSize);
            case 'blackman'
                analyWindow = blackman_local(fftSize);
            case 'sine'
                analyWindow = sine_local(fftSize);
            otherwise
                error('Input window type is not supported. Type "help STFT" and check options.\n')
        end
    end
end

% Pad zeros before and after signal values
nch = size(signal,2); % number of channels
zeroPadSize = fftSize - shiftSize; % size of zero padding
signal = [zeros(zeroPadSize,nch); signal; zeros(fftSize,nch)]; % padding zeros
length = size(signal,1); % zero-padded signal length

% Calculate STFT
nframes = floor( (length - fftSize + shiftSize) / shiftSize ); % number of frames in spectrogram
spectrogram = zeros( fftSize/2+1, nframes, nch ); % memory allocation (freq. x nframe x nch)
for ch = 1:nch
    for n = 1:nframes
        startPoint = (n-1)*shiftSize; % start point of windowing
        spectrum = fft( signal(startPoint+1:startPoint+fftSize,ch) .* analyWindow ); % fft spectrum of windowed short-time signal
        spectrogram(:,n,ch) = spectrum( 1:fftSize/2+1, 1 ); % substitute spectrum (only 0Hz to Nyquist frequency components)
    end
end
end

%% Local functions
function analyWindow = hamming_local(fftSize)
t = linspace(0,1,fftSize+1).'; % periodic (produce L+1 window and return L window)
analyWindow = 0.54*ones(fftSize,1) - 0.46*cos(2.0*pi*t(1:fftSize));
end

function analyWindow = hann_local(fftSize)
t = linspace(0,1,fftSize+1).'; % periodic (produce L+1 window and return L window)
analyWindow = max(0.5*ones(fftSize,1) - 0.5*cos(2.0*pi*t(1:fftSize)),eps);
end

function analyWindow = rectangular_local(fftSize)
analyWindow = ones(fftSize,1);
end

function analyWindow = blackman_local(fftSize)
t = linspace(0,1,fftSize+1).'; % periodic (produce L+1 window and return L window)
analyWindow = max(0.42*ones(fftSize,1) - 0.5*cos(2.0*pi*t(1:fftSize)) + 0.08*cos(4.0*pi*t(1:fftSize)),eps);
end

function analyWindow = sine_local(fftSize)
t = linspace(0,1,fftSize+1).'; % periodic (produce L+1 window and return L window)
analyWindow = max(sin(pi*t(1:fftSize)),eps);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%