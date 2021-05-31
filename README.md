# Independent Low-Rank Matrix Analysis (ILRMA)

## About
Sample MATLAB script for independent low-rank matrix analysis (ILRMA) and its application to blind audio source separation.

## Contents
- input [dir]:		        includes test audio signals (reverberation time is around 300 ms)
- reference [dir]:	        includes reference documents for ILRMA
- consistentILRMA.m:        blind source separation based on consistent ILRMA
- ILRMA.m:			        blind source separation based on ILRMA
- ILRMAISS.m                blind source separation based on ILRMA with iterative source steering update
- ISTFT.m:			        inverse short-time Fourier transform
- main.m:			        main script with parameter settings
- backProjection.m:	        back projection technique (fixing frequency-wise scales)
- STFT.m:			        short-time Fourier transform
- whitening.m:		        applying principal component analysis for decorrelating observed multichannel signal

## Usage Note
A parameter "applyNormalize" in 47th line of main.m is important. 
This parameter controls whether apply normalization process in each iteration of ILRMA.  
Normalization process will improve numerical stability of the algorithm, but the monotonic decrease of the cost function in the update algorithm may be lost.  
Useful information can be found at http://d-kitamura.net/pdf/misc/AlgorithmsForIndependentLowRankMatrixAnalysis.pdf

## Copyright Note
Copyright 2021 Daichi Kitamura.  
These programs are distributed only for academic research at universities and research institutions.  
It is not allowed to use or modify these programs for commercial or industrial purpose without our permission.  
When you use or modify these programs and write research articles, cite the following references: 
* D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined blind source separation unifying independent vector analysis and nonnegative matrix factorization," IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, September 2016.
* D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined blind source separation with independent low-rank matrix analysis," Audio Source Separation. Signals and Communication Technology., S. Makino, Ed. Springer, Cham, pp. 125-155, March 2018.
* D. Kitamura and K. Yatabe, "Consistent independent low-rank matrix analysis for determined blind source separation," EURASIP J. Adv. Signal Process., vol. 2020, no. 46, p. 35, November 2020.

## References
* N. Ono, "Stable and fast update rules for independent vector analysis based on auxiliary function technique", IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), pp.189-192, 2011.
* S. Robin and N. Ono, "Fast and stable blind source separation with rank-1 updates", IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp.236-240, 2020.
* D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined blind source separation unifying independent vector analysis and nonnegative matrix factorization," IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, September 2016.
* D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined blind source separation with independent low-rank matrix analysis," Audio Source Separation. Signals and Communication Technology., S. Makino, Ed. Springer, Cham, pp. 125-155, March 2018.
* D. Kitamura and K. Yatabe, "Consistent independent low-rank matrix analysis for determined blind source separation," EURASIP J. Adv. Signal Process., vol. 2020, no. 46, p. 35, November 2020.

## Python Script
You can find Python script of ILRMA in Pyroomacoustics: https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.bss.ilrma.html#module-pyroomacoustics.bss.ilrma

## See Also
* HP: http://d-kitamura.net
* Demo: http://d-kitamura.net/demo-ILRMA_en.html