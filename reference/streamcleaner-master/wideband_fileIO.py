#This program is a tentative experiment wideband 2MHz IQ processing offline.
'''
FILE wideband_fileIO.py 0.1
Copyright 2022 Oscar Steila, Joshuah Rainstar
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
'''
Copyright 2022 Oscar Steila, Joshuah Rainstar

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''
#how to test with HDSDR and ExtIOsddc using BBRF103, RX888... 
#Setup the ExtIO output sampling rate to HDSR to 2MHz complex float32 stream (select IF bandwidth 2MHz)
#Record same seconds (10) of RF input signal using HDSDR RF recorder that outputs "RFfilename.wav".
#the file must tune the LO to the carrier frequency of the target signal.
#process the "RFfilename.wav" to produce "cleanup, RFfilename.wav" with this code using Mambaforge-Windows-x86_64 

#How to use this file:
#step one: using https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
#install python.
#step three: locate the dedicated python terminal in your start menu, called mambaforge prompt.
#within that prompt, give the following instructions:
#conda update --all
#conda install pip numpy scipy
#pip install tk numba pyroomacoustics
#if all of these steps successfully complete, you're ready to go, otherwise fix things.
#run it by executing this command in the terminal, in the folder containing the file: python wideband_fileIO.py

#Now playback the cleanup file in HDSDR and demodulate it with HDSDR tuned to 0.0Hz as 
#the wavefile procedure discharges the HDSDR chunk with frequency information
#

from scipy.io import wavfile
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time

import os
import numba
import numpy
import pyroomacoustics as pra
from threading import Thread
import pyaudio

# parameters 
FFTN = 16384 # frame time =  16384/2000000 => 8.192 ms ; 122 Hz bin
N_BINS = int((FFTN * 3400)/2000000) # input sampling rate = 2000000 Hz bandwidth 3400 Hz => NBINS = 27
N_HOP = int(FFTN/4) 

@numba.njit(numba.float64(numba.float64[:]))
def man(arr):
    med = numpy.nanmedian(arr[numpy.nonzero(arr)])
    return numpy.nanmedian(numpy.abs(arr - med))

@numba.njit(numba.float64(numba.float64[:]))
def atd(arr):
    x = numpy.square(numpy.abs(arr - man(arr)))
    return numpy.sqrt(numpy.nanmean(x))

@numba.njit(numba.float64(numba.float64[:]))
def threshold(data: numpy.ndarray):
 a = numpy.sqrt(numpy.nanmean(numpy.square(numpy.abs(data -numpy.nanmedian(numpy.abs(data - numpy.nanmedian(data[numpy.nonzero(data)]))))))) + numpy.nanmedian(data[numpy.nonzero(data)])
 return a

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'same') / w

def smoothpadded(data: numpy.ndarray,n:int):
  o = numpy.pad(data, n*2, mode='median')
  return moving_average(o,n)[n*2: -n*2]

def numpy_convolve_filter_longways(data: numpy.ndarray,N:int,M:int):
  E = N*2
  d = numpy.pad(array=data,pad_width=((0,0),(E,E)),mode="constant")  
  b = numpy.ravel(d)  
  for all in range(M):
       b[:] = ( b[:]  + (numpy.convolve(b[:], numpy.ones(N),mode="same") / N)[:])/2
  return d[:,E:-E]

def numpy_convolve_filter_topways(data: numpy.ndarray,N:int,M:int):
  E = N*2
  d = numpy.pad(array=data,pad_width=((E,E),(0,0)),mode="constant")  
  d = d.T.copy()  
  b = numpy.ravel(d)  
  for all in range(M):
       b[:] = ( b[:]  + (numpy.convolve(b[:], numpy.ones(N),mode="same")[:] / N)[:])/2
  d = d.T
  return d[E:-E:]


def generate_true_logistic(points):
    fprint = numpy.linspace(0.0,1.0,points)
    fprint[1:-1]  /= 1 - fprint[1:-1]
    fprint[1:-1]  = numpy.log(fprint[1:-1])
    fprint[-1] = ((2*fprint[-2])  - fprint[-3]) 
    fprint[0] = -fprint[-1]
    return numpy.interp(fprint, (fprint[0], fprint[-1]),  (0, 1))

def generate_logit_window(size,sym =True):
  if sym == False or size<32:
    print("not supported")
    return numpy.zeros(size)
  if size % 2:
    e = generate_true_logistic((size+1)//2)
    result = numpy.zeros(size)
    result[0:(size+1)//2] = e
    result[(size+1)//2:] = e[0:-1][::-1]
    return result
  else:
    e = generate_true_logistic(size//2)
    e = numpy.hstack((e,e[::-1]))
    return e

@numba.jit()
def generate_hann(M, sym=True):
    a = [0.5, 0.5]
    fac = numpy.linspace(-numpy.pi, numpy.pi, M)
    w = numpy.zeros(M)
    for k in range(len(a)):
        w += a[k] * numpy.cos(k * fac)
    return w

def logit_generation(npoints):
        logit = numpy.linspace(0, 1, npoints, dtype=numpy.float64)
        logit[1:-1] /= 1 - logit[1:-1]       # all but first and last, divide by 1 - the value.
        logit[1:-1] = numpy.log(logit[1:-1])  # all but the first and last, log(value)
        logit[-1] = ((2 * logit[-2]) - logit[-3])
        logit[0] = -logit[-1]
        m = logit[-1]
        logit += m
        logit /= m * 2
        return logit
#note: if you alter the number of bins, you need to regenerate this array. currently set to consider N_BINS bins        
LOGIT = logit_generation(N_BINS)  

@numba.njit(numba.float64[:](numba.float64[:,:]))
def fast_entropy(data: numpy.ndarray):
   entropy = numpy.zeros(data.shape[1],dtype=numpy.float64)
   for each in numba.prange(data.shape[1]):
      d = data[:,each]
      d = numpy.interp(d, (d[0], d[-1]), (0, +1))
      entropy[each] = 1 - numpy.corrcoef(d, LOGIT)[0,1]
   return entropy


@numba.jit(numba.float64[:,:](numba.float64[:,:],numba.int32[:],numba.float64,numba.float64[:]))
def fast_peaks(stft_:numpy.ndarray,entropy:numpy.ndarray,thresh:numpy.float32,entropy_unmasked:numpy.ndarray):
    #0.01811 practical lowest - but the absolute lowest is 0. 0 is a perfect logistic. 
    #0.595844362 practical highest - an array of zeros with 1 value of 1.
    #note: this calculation is dependent on the size of the logistic array.
    #if you change the number of bins, this constraint will no longer hold and renormalizing will fail.
    #a lookup table is not provided, however the value can be derived by setting up a one-shot fast_entropy that only considers one array,
    #sorting, interpolating, and comparing it. 
    mask = numpy.zeros_like(stft_)
    for each in numba.prange(stft_.shape[1]):
        data = stft_[:,each].copy()
        if entropy[each] == 0:
            mask[0:N_BINS,each] =  0
            continue #skip the calculations for this row, it's masked already
        constant = atd(data) + man(data)  #by inlining the calls higher in the function, it only ever sees arrays of one size and shape, which optimizes the code
        test = entropy_unmasked[each]  / 0.6091672572096941 #currently set for N_BINS bins
        test = abs(test - 1) 
        thresh1 = (thresh*test)
        if numpy.isnan(thresh1):
            thresh1 = constant #catch errors
        constant = (thresh1+constant)/2
        data[data<constant] = 0
        data[data>0] = 1
        mask[0:N_BINS,each] = data[:]
    return mask

@numba.njit(numba.float64(numba.float64[:]))
def threshhold(arr):
  return (atd(arr)+ numpy.nanmedian(arr[numpy.nonzero(arr)])) 

@numba.njit(numba.int32(numba.int32[:]))
def longestConsecutive(nums: numpy.ndarray):
        longest_streak = 0
        streak = 0
        prevstreak = 0
        for num in range(nums.size):
          if nums[num] == 1:
            streak += 1
          if nums[num] == 0:
            prevstreak = max(streak,prevstreak)
            streak = 0
        return max(streak,prevstreak)
    

import copy

def mask_generate(stft_vh1: numpy.ndarray,stft_vl1: numpy.ndarray):

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this. let's just go with N_BINS bins.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.
    stft_vh = numpy.ndarray(shape=stft_vh1.shape, dtype=numpy.float64, order='C') 
    stft_vl = numpy.ndarray(shape=stft_vh1.shape, dtype=numpy.float64,order='C') 
    stft_vh[:] = copy.deepcopy(stft_vh1)
    stft_vl[:] = copy.deepcopy(stft_vl1)

    #24000/256 = 93.75 hz per frequency bin.
    #a 4000 hz window(the largest for SSB is roughly 43 bins.
    #https://en.wikipedia.org/wiki/Voice_frequency
    #however, practically speaking, voice frequency cuts off just above 3400hz.
    #*most* SSB channels are constrained far below this.
    #to catch most voice activity on shortwave, we use the first 32 bins, or 3000hz.
    #we automatically set all other bins to the residue value.
    #reconstruction or upsampling of this reduced bandwidth signal is a different problem we dont solve here.

    lettuce_euler_macaroni = 0.057 #was grossman constant but that was arbitrarily chosen
    stft_vs = numpy.sort(stft_vl[0:N_BINS,:],axis=0) #sort the array
    
    entropy_unmasked = fast_entropy(stft_vs)
    entropy = smoothpadded(entropy_unmasked,14)
    factor = numpy.max(entropy)
    
    if factor < lettuce_euler_macaroni: 
      return stft_vh.T * 1e-6

    entropy[entropy<lettuce_euler_macaroni] = 0
    entropy[entropy>0] = 1
    entropy = entropy.astype(dtype=numpy.int32)
    nbins = numpy.sum(entropy)
    maxstreak = longestConsecutive(entropy)

    #ok, so the shortest vowels are still over 100ms long. That's around 37.59 samples. Call it 38.
    #half of 38 is, go figure, 17.
    #now we do have an issue- what if a vowel is being pronounced in the middle of transition?
    #this *could* result in the end or beginning of words being truncated, but with this criteria,
    #we reasonably establish that there are no regions as long as half a vowel.
    #if there's really messed up speech(hence small segments) but enough of it(hence 22 bins)
    #then we can consider the frame to consist of speech
    # an ionosound sweep is also around or better than 24 samples, also
    # number of bins and streak doubled for hop size 64
    # for hop size = N_HOP, nbins = 22, maxstreak = 16
    #if hop size 256 was chosen, nbins would be 11 and maxstreak would be 11.
    #however, fs/hop = maximum alias-free frequency. For hop size of 64, it's over 300hz.
    if nbins<22 and maxstreak<16:
      return stft_vh.T * 1e-6
    mask=numpy.zeros_like(stft_vh)

    stft_vh1 = stft_vh[0:N_BINS,:]
    thresh = threshold(stft_vh1[stft_vh1>man(stft_vl[0:N_BINS,:].flatten())])/2

    mask[0:N_BINS,:] = fast_peaks(stft_vh[0:N_BINS,:],entropy,thresh,entropy_unmasked)
    mask = numpy_convolve_filter_longways(mask,5,17)
    mask2 = numpy_convolve_filter_topways(mask,5,2)
    mask2 = numpy.where(mask==0,0,mask2)
    mask2 = (mask2 - numpy.nanmin(mask2)) / numpy.ptp(mask2) #normalize to 1.0
    mask2[mask2<1e-6] = 1e-6 #backfill the residual
    return mask2.T


class FilterThread(Thread):
    def __init__(self):
        super(FilterThread, self).__init__()
        self.running = True
        self.NFFT = FFTN 
        self.NBINS=32
        self.hop = N_HOP
        self.hann = generate_hann(FFTN)
        self.logit = generate_logit_window(FFTN)
        self.synthesis = pra.transform.stft.compute_synthesis_window(self.hann, self.hop)
        self.stft = pra.transform.STFT(N=FFTN, hop=self.hop, analysis_window=self.hann,synthesis_window=self.synthesis ,online=True)
        self.stftl = pra.transform.STFT(N=FFTN, hop=self.hop, analysis_window=self.logit,synthesis_window=self.synthesis ,online=True)

    def process(self,data):         
           self.stft.analysis(data) #generate our complex representation
           self.stftl.analysis(data)
           mask = mask_generate(numpy.abs(self.stft.X.T),numpy.abs(self.stftl.X.T))
           output = self.stft.synthesis(X=self.stft.X* mask)
           return output
    def stop(self):
        self.running = False 

        
def padarray(A, size):
    t = size - len(A)
    return numpy.pad(A, pad_width=(0, t), mode='constant',constant_values=numpy.std(A))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process_data(data: numpy.ndarray):
    print("processing ", data.size / rate, " seconds long file at ", rate, " rate.")
    start = time.time()
    filter = FilterThread()
    processed = []
    for each in chunks(data, rate):
        if each.size == rate:
            processed.append(filter.process(each))
        else:
            psize = each.size
            working = padarray(each, rate)
            processed.append(filter.process(working)[0:psize])
    end = time.time()
    print("took ", end - start, " to process ", data.size / rate)
    return numpy.concatenate((processed), axis=0).astype(dtype=data.dtype)           


if __name__ == '__main__':
# look for input file.wav
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(filetypes=(('wav files', '*.wav'),), title='Select input file.wav')
    file_path, file_tail = os.path.split(filename)
    infile = filename
    outfile = f"{file_path}/cleanup{FFTN}_{N_BINS}_{file_tail}"
    print(infile)
    print(outfile)
    print(f'FFT {FFTN}')
    print(f'BINS {N_BINS}')
    rate, data = wavfile.read(infile)
    ndim = len(data.shape)
    print( f'input data {ndim}D' )
    if ndim == 1:
        data = numpy.asarray(data)#correct the format for processing
        reduced_noise = process_data(data)     
    else:  # stereo 
        reduced_noise = numpy.column_stack((process_data(data[:,0]), process_data(data[:,1]))) 
    wavfile.write(outfile, rate, reduced_noise)     
