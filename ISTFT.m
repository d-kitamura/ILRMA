function waveform = ISTFT(spectrogram, shiftSize, window, orgLength)
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
%   waveform = ISTFT(spectrogram)
%   waveform = ISTFT(spectrogram,shiftSize)
%   waveform = ISTFT(spectrogram,shiftSize,window)
%   waveform = ISTFT(spectrogram,shiftSize,window,orgLength)
%
% [inputs]
%  spectrogram: STFT of input signal (frequency bins (fftSize/2+1) x time frames x channels)
%    shiftSize: frame shift length (default: fftSize/2)
%       window: analysis window function used in STFT (fftSize x 1) or choose used analysis window function from below:
%               'hamming'    : Hamming window (default)
%               'hann'       : von Hann window
%               'rectangular': rectangular window
%               'blackman'   : Blackman window
%               'sine'       : sine window
%    orgLength: length of original signal (before zero padding) (default: the same as that of output signal)
%
% [outputs]
%   waveform: time-domain waveform of input spectrogram (signal x channels)
%

% Check errors and set default values
if (nargin < 1)
    error('Too few input arguments.\n');
end
[nfreqs, nframes, nch] = size(spectrogram);
if (nch > nfreqs)
    error('Input spectrogram might be wrong. The size of it must be (freq x frame x ch).\n');
end
if (isreal(spectrogram) == 1)
    error('Input spectrogram might be wrong. It does not complex-valued matrix.\n');
end
if (mod(nfreqs,2) == 0)
    error('The number of rows of sectrogram must be an odd number because it is (fftSize/2)+1.\n');
end
fftSize = (nfreqs-1) * 2; % fft length used in STFT
if (nargin < 2)
    shiftSize = fftSize/2;
elseif (mod(fftSize,shiftSize) ~= 0)
    error('fftSize must be dividable by shiftSize.\n');
end
if (nargin<3)
    analyWindow = hamming_local(fftSize); % default window
else
    if (isnumeric(window))
        if (size(window, 1) ~= fftSize)
            error('The length of synthesis window must be the same as fftSize used in STFT.\n');
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
                error('Input window type is not supported. Type "help ISTFT" and check options.\n')
        end
    end
end

% Calculate optimal synthesis window based on minimal distortion principle
synthWindow = optSynthWnd_local( analyWindow, shiftSize );

% Inverse STFT
spectrum = zeros(fftSize,1); % memory allocation
tmpSignal = zeros((nframes-1)*shiftSize+fftSize,nch); % memory allocation (zero-padded signal length x nch)
for ch = 1:nch
    for n = 1:nframes
        spectrum(1:fftSize/2+1,1) = spectrogram(:,n,ch);
        spectrum(1,1) = spectrum(1,1)/2; % 0Hz component
        spectrum(fftSize/2+1,1) = spectrum(fftSize/2+1,1)/2; % Nyquist frequency component
        startPoint = (n-1)*shiftSize;
        tmpSignal(startPoint+1:startPoint+fftSize,ch) = tmpSignal(startPoint+1:startPoint+fftSize,ch) + real(ifft(spectrum,fftSize).*synthWindow)*2;
    end
end
waveform = tmpSignal(fftSize-shiftSize+1:(nframes-1)*shiftSize+fftSize,:); % discard padded zeros in pre-processing of STFT

% Discarding padded zeros in the end of the signal
if (nargin==4)
    waveform = waveform(1:orgLength,:);
end
end

%% Local functions
function synthWindow = optSynthWnd_local(analyWindow,shiftSize) % based on minimal distortion principle
fftSize = size(analyWindow,1);
synthWindow = zeros(fftSize,1);
for i = 1:shiftSize
    amp = 0;
    for j = 1:fftSize/shiftSize
        amp = amp + analyWindow(i+(j-1)*shiftSize,1)*analyWindow(i+(j-1)*shiftSize,1);
    end
    for j = 1:fftSize/shiftSize
        synthWindow(i+(j-1)*shiftSize,1) = analyWindow(i+(j-1)*shiftSize,1)/amp;
    end
end
end

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