function [specgram,analyWin,sigLen] = STFT(sig,fftSize,shiftSize,winType)
%
% Short-time Fourier transform
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% See also:
% http://d-kitamura.net
%
% [syntax]
%   [specgram,analyWin,sigLen] = STFT(sig)
%   [specgram,analyWin,sigLen] = STFT(sig,fftSize)
%   [specgram,analyWin,sigLen] = STFT(sig,fftSize,shiftSize)
%   [specgram,analyWin,sigLen] = STFT(sig,fftSize,shiftSize,winType)
%
% [inputs]
%          sig: input signal (length x channels)
%      fftSize: window length [points] in STFT (scalar, even number, default: 1024)
%    shiftSize: shift length [points] in STFT (scalar, default: fftSize/2)
%      winType: window function used in STFT (name of window function, default: 'hamming')
%               'hamming'    : Hamming window (default)
%               'hann'       : von Hann window
%               'rectangular': rectangular window
%               'blackman'   : Blackman window
%               'sine'       : sine window
%
% [outputs]
%     specgram: spectrogram of input signal (frequency bins (fftSize/2+1) x time frames x channels)
%     analyWin: analysis window function used in STFT (fftSize x 1) and can be used for calculating optimal synthesis window
%       sigLen: length of original signal without zero padding
%

% Arguments check and set default values
arguments
    sig (:,:) double
    fftSize (1,1) double {mustBeInteger(fftSize)} = 1024
    shiftSize (1,1) double {mustBeInteger(shiftSize)} = fftSize/2
    winType char {mustBeMember(winType,{'hamming','hann','rectangular','blackman','sine'})} = 'hamming'
end

% Errors check
[sigLen, nCh] = size(sig); % get signal length and number of channels
if sigLen < nCh; error('The size of input signal might be wrong. The signal must be length x channels size.\n'); end
if mod(fftSize,2) ~= 0; error('fftSize must be an even number.\n'); end
if mod(fftSize,shiftSize) ~= 0; error('fftSize must be dividable by shiftSize.\n'); end
switch winType
    case 'hamming'; analyWin = local_hamming(fftSize);
    case 'hann'; analyWin = local_hann(fftSize);
    case 'rectangular'; analyWin = local_rectangular(fftSize);
    case 'blackman'; analyWin = local_blackman(fftSize);
    case 'sine'; analyWin = local_sine(fftSize);
    otherwise; error('Input winType is not supported. Type "help STFT" and check options.\n');
end

% Pad zeros at the beginning and ending of the input signal
zeroPadSize = fftSize - shiftSize; % size of zero padding
padSig = [zeros(zeroPadSize,nCh); sig; zeros(fftSize,nCh)]; % padding zeros
padSigLen = size(padSig,1); % zero-padded signal length

% Calculate STFT
nFrame = floor((padSigLen - fftSize + shiftSize) / shiftSize); % number of time frames in spectrogram
specgram = zeros(fftSize/2+1, nFrame, nCh); % memory allocation (nFreq x nFrames x nCh)
shortTimeSig = zeros(fftSize, nFrame); % memory allocation (nFreq x nFrames x nCh)
for iCh = 1:nCh
    for iFrame = 1:nFrame % get short-time signals by framing
        startPoint = (iFrame-1)*shiftSize; % start point of short-time signal
        shortTimeSig(:,iFrame) = padSig(startPoint+1:startPoint+fftSize, iCh); % store short-time signal
    end
    tmp = fft(shortTimeSig .* analyWin); % get DFT spectra of windowed short-time signals
    specgram(:,:,iCh) = tmp(1:fftSize/2+1, :); % store spectrum (only from DC to Nyquist frequency components)
end
end

%% Local functions
function win = local_hamming(fftSize)
t = linspace(0, 1, fftSize+1).'; % periodic (produce L+1 window and return L window)
win = 0.54*ones(fftSize,1) - 0.46*cos(2.0*pi*t(1:fftSize));
end

function win = local_hann(fftSize)
t = linspace(0, 1, fftSize+1).'; % periodic (produce L+1 window and return L window)
win = max(0.5*ones(fftSize,1) - 0.5*cos(2.0*pi*t(1:fftSize)),eps);
end

function win = local_rectangular(fftSize)
win = ones(fftSize,1);
end

function win = local_blackman(fftSize)
t = linspace(0, 1,fftSize+1).'; % periodic (produce L+1 window and return L window)
win = max(0.42*ones(fftSize,1) - 0.5*cos(2.0*pi*t(1:fftSize)) + 0.08*cos(4.0*pi*t(1:fftSize)),eps);
end

function win = local_sine(fftSize)
t = linspace(0, 1, fftSize+1).'; % periodic (produce L+1 window and return L window)
win = max(sin(pi*t(1:fftSize)),eps);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%