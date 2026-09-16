# Intrinsic Bayesian Algorithm for Shortwave Voice Noise Reduction(IBA-SVNR):
This repository contains a really good algorithm for reducing shortwave noise in audio, for use with speech and other similar waveforms.
Note that this algorithm should not be used for denoising data modes.

It is free to use, and the code is MIT/GPL licensed. 
It works on the basis of bayesian estimation and thresholding using robust measures.
It is faster than realtime and coded to be optimal for speed, at most, O log n, typically closer to O(n).

It is meant for use on shortwave radio, with bandlimited single sideband signals with bandwidth of 4000hz or less.

The algorithm was coded in python and realtime.py is the reference with the best performance.
The C++ was derived from the python and offers good performance.

The vst and lv2 are compiled for windows, but upon special request can be provided for other platforms.

for linux, try https://github.com/robbert-vdh/yabridge
to get the full benefits without adding a lot of garbage to your workstation:
https://sourceforge.net/projects/equalizerapo/ this will add an audio effect (like bass boost, virtual surround, etc) that accepts vsts.
it doesnt accept vst3, so it cant be used for now but maybe someday.
for now , https://www.hermannseib.com/english/vsthost.htm vsthost is an excellent program I have enjoyed using a long time. 
you can use this https://vb-audio.com/Cable/index.htm virtual audio cable:
route your SDR audio -> virtual audio -> vsthost (load the VST) -> your speakers.


Thanks and credit should be given to the following people:
John Muradeli OverLordGoldDragon,
the Pyroomacoustics team,
The Numpy team,
my friends on discord,
Chatgpt(because it basically did for me what millions of engineers refused to- answer my many questions, instead of 
telling me to "go read a book")



