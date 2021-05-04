function sig = ISTFT(specgram,shiftSize,analyWin,orgLen)
%
% Inverse short-time Fourier transform
% Synthesis window is calculated based on minimal distortion principle,
% which is described below:
% D. Griffin and J. Lim, "Signal estimation from modified short-time
% Fourier transform," IEEE Transactions on Acoustics, Speech, and Signal
% Processing, vol. 32, no. 2, pp. 236-243, 1984.
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% See also:
% http://d-kitamura.net
%
% [syntax]
%   sig = ISTFT(specgram,shiftSize)
%   sig = ISTFT(specgram,shiftSize,analyWin)
%   sig = ISTFT(specgram,shiftSize,analyWin,orgLen)
%
% [inputs]
%     specgram: STFT of input signal (frequency bins (fftSize/2+1) x time frames x channels)
%    shiftSize: frame shift length
%     analyWin: analysis window function used in STFT (fftSize x 1) or choose used analysis window function from below:
%               'hamming'    : Hamming window (default)
%               'hann'       : von Hann window
%               'rectangular': rectangular window
%               'blackman'   : Blackman window
%               'sine'       : sine window
%       orgLen: length of original signal (before zero padding) (default: the same as that of output signal)
%
% [outputs]
%          sig: time-domain waveform of input spectrogram (signal x channels)
%

% Arguments check and set default values
arguments
    specgram (:,:,:) {mustBeNumeric}
    shiftSize (1,1) double {mustBeInteger(shiftSize)}
    analyWin
    orgLen (1,1) double {mustBeInteger(orgLen)}
end

% Error check
[nFreq, nFrame, nCh] = size(specgram);
fftSize = (nFreq-1) * 2; % fft length used in STFT
if nCh > nFreq; error('Input spectrogram might be wrong. The size of it must be (freq x frame x ch).\n'); end
if isreal(specgram); error('Input spectrogram might be wrong. It does not complex-valued matrix.\n'); end
if mod(nFreq,2) == 0; error('The number of rows of sectrogram must be an odd number because it is (fftSize/2)+1.\n'); end
if mod(fftSize,shiftSize) ~= 0; error('fftSize must be dividable by shiftSize.\n'); end
if nargin < 3
    analyWin = local_hamming(fftSize); % default window
else
    if isnumeric(analyWin)
        if size(analyWin, 1) ~= fftSize; error('The length of synthesis window must be the same as fftSize used in STFT.\n'); end
    else
        switch analyWin
            case 'hamming'; analyWin = local_hamming(fftSize);
            case 'hann'; analyWin = local_hann(fftSize);
            case 'rectangular'; analyWin = local_rectangular(fftSize);
            case 'blackman'; analyWin = local_blackman(fftSize);
            case 'sine'; analyWin = local_sine(fftSize);
            otherwise; error('Input window type is not supported. Type "help ISTFT" and check options.\n');
        end
    end
end

% Calculate optimal synthesis window based on minimal distortion principle
synthWin = local_optSynthWin(analyWin, shiftSize);

% Inverse STFT
tmpSig = zeros((nFrame-1)*shiftSize+fftSize, nCh); % memory allocation (zero-padded signal, length x nch)
specgram(1,:,:) = specgram(1,:,:)/2; % DC component
specgram(fftSize/2+1,:,:) = specgram(fftSize/2+1,:,:)/2; % Nyquist frequency component
for iCh = 1:nCh
    shortTimeSig = real(ifft(specgram(:,:,iCh), fftSize) .* synthWin) * 2;
    for iFrame = 1:nFrame % overlap add of short-time signals
        startPoint = (iFrame-1)*shiftSize;
        tmpSig(startPoint+1:startPoint+fftSize,iCh) = tmpSig(startPoint+1:startPoint+fftSize,iCh) + shortTimeSig(:,iFrame);
    end
end
sig = tmpSig(fftSize-shiftSize+1:(nFrame-1)*shiftSize+fftSize, :); % discard padded zeros at beginning of signal, which are added in STFT

% Discarding padded zeros at the end of the signal
if exist('orgLen', 'var')
    sig = sig(1:orgLen,:);
end
end

%% Local functions
function synthWin = local_optSynthWin(analyWin,shiftSize) % based on minimal distortion principle
fftSize = size(analyWin,1);
synthWin = zeros(fftSize,1);
for i = 1:shiftSize
    amp = 0;
    for j = 1:fftSize/shiftSize
        amp = amp + analyWin(i+(j-1)*shiftSize,1)*analyWin(i+(j-1)*shiftSize,1);
    end
    for j = 1:fftSize/shiftSize
        synthWin(i+(j-1)*shiftSize,1) = analyWin(i+(j-1)*shiftSize,1)/amp;
    end
end
end

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
t = linspace(0, 1, fftSize+1).'; % periodic (produce L+1 window and return L window)
win = max(0.42*ones(fftSize,1) - 0.5*cos(2.0*pi*t(1:fftSize)) + 0.08*cos(4.0*pi*t(1:fftSize)),eps);
end

function win = local_sine(fftSize)
t = linspace(0, 1, fftSize+1).'; % periodic (produce L+1 window and return L window)
win = max(sin(pi*t(1:fftSize)),eps);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%