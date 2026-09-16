'''
Copyright 2023 Joshuah Rainstar
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
'''
Copyright 2023 Joshuah Rainstar
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
#tags: stft, istft, rfft, irfft - all in numba- all competitive with numpy and other libraries- all mostly correct and generate output equivalent to numpy.
#Note: please do not use this code if you dont understand what it's doing. i could not have done this without the help of chatgpt 4.0.
#very little documentation which is legible to the commoner on how to write these functions is out there, here is the code and i hope it instructs you.
#Every effort has been made to ensure it works correctly, but it was not written as an adaptable function designed for a variety of inputs.
#included here are two versions each of rFFT and irFFT, to create a diversity of solutions that best help explain the algorithm.
#The error rate here is less than 1e-13, so as long as you're working with well behaved inputs, there should not be a problem.
#Consult https://github.com/ChangWeiTan/MultiRocket/blob/3ccaa4f8ed11769b904af885e17018c39c89a2ac/utils/tools.py#L10
#and https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/ to find out more

import numba
import numpy as np

tw = [np.exp(-1.0 * 1.0j * np.pi * np.arange(((2**i)/2), dtype=np.complex256) / ((2**i)/2)) for i in range(1,11)]
list_of_lists = [list(map(lambda x: [x.astype(np.complex128)], arr)) for arr in tw]
twiddlefactors = np.concatenate(list_of_lists)
inverse = np.conj(twiddlefactors) #use this for irfft

@numba.jit(numba.complex128[:](numba.float64[:],numba.complex128[:,:]),fastmath=True,nopython=True)
def unrolled_numba_rfft(input_data:np.ndarray, twiddlefactors: np.ndarray):
    #input must be float64, exactly 512 elements. Outputs 257 elements, complex128.
    #this function has been constrained to only work on real data(float64), power of two arrays of size 512, and only returns 257 elements.
    #Cooley-Tukey radix-2 Decimation in Time unrolled loop, precomputed twiddlefactors



    ## Initialization Section: This part of the code sets up the matrix required for "Butterfly" operations in FFT (Fast Fourier Transform). 
    #It then initializes arrays for storing the result at each stage of the FFT calculation. 
    #Each array represents a stage in the FFT computation, with the dimensions set according to the 'radix-2' nature of the calculation. T
    #This means that at each stage, the data is divided in half.
    
  
    M = np.asarray([[1.+0.0000000e+00j, 1.+0.0000000e+00j],[ 1.+0.0000000e+00j, -1.-1.2246468e-16j]],dtype=numpy.complex128) #butterfly matrix

    X_stage_0 = np.zeros((2, 256), dtype=np.complex128)
    X_stage_1 = np.zeros((4, 128), dtype=np.complex128)
    X_stage_2 = np.zeros((8, 64), dtype=np.complex128)
    X_stage_3 = np.zeros((16, 32), dtype=np.complex128)
    X_stage_4 = np.zeros((32, 16), dtype=np.complex128)
    X_stage_5 = np.zeros((64, 8), dtype=np.complex128)
    X_stage_6 = np.zeros((128, 4), dtype=np.complex128)
    X_stage_7 = np.zeros((256, 2), dtype=np.complex128)
    X_stage_8 = np.zeros((512, 1), dtype=np.complex128)

    #Stage-0 FFT Calculation: In this section, the code computes the first stage of the FFT calculation. 
    #It performs "Butterfly" operations on the input data using the previously initialized matrix.
    #A "Butterfly" operation involves a simple pattern of multiplications and additions between pairs of input data points.
    
    X_stage_0[0, :] = M[0, 0] * input_data[:256]  + M[0, 1] * input_data[256:]
    X_stage_0[1, :] = M[1, 0] * input_data[:256]  + M[1, 1] * input_data[256:]
    
    # Multi-Stage FFT Calculations: This section of the code computes the rest of the stages in the FFT calculation. 
    #At each stage, the code computes the "Butterfly" operations, now involving the 'twiddle factors' -
    #complex roots of unity that are integral to the FFT algorithm. 
    #Each stage reduces the data's size by half and computes the FFT of these smaller chunks.




    e = 2 #twiddle_index
    q = 128 #quarter length
    X_stage_1[:e, :q] = X_stage_0[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_0[:e, q:2*q]
    X_stage_1[e:e*2, :q] = X_stage_0[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_0[:e, q:2*q]

    e = 4
    q = 64
    X_stage_2[:e, :q] = X_stage_1[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_1[:e, q:2*q]
    X_stage_2[e:e*2, :q] = X_stage_1[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_1[:e, q:2*q]

    e = 8
    q = 32
    X_stage_3[:e, :q] = X_stage_2[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_2[:e, q:2*q]
    X_stage_3[e:e*2, :q] = X_stage_2[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_2[:e, q:2*q]
    
    e = 16
    q = 16
    X_stage_4[:e, :q] = X_stage_3[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_3[:e, q:2*q]
    X_stage_4[e:e*2, :q] = X_stage_3[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_3[:e, q:2*q]

    e = 32
    q = 8
    X_stage_5[:e, :q] = X_stage_4[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_4[:e, q:2*q]
    X_stage_5[e:e*2, :q] = X_stage_4[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_4[:e, q:2*q]

    e = 64
    q = 4
    X_stage_6[:e, :q] = X_stage_5[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_5[:e, q:2*q]
    X_stage_6[e:e*2, :q] = X_stage_5[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_5[:e, q:2*q]

    e = 128
    q = 2
    X_stage_7[:e, :q] = X_stage_6[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_6[:e, q:2*q]
    X_stage_7[e:e*2, :q] = X_stage_6[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_6[:e, q:2*q]

    e = 256
    q = 1
    X_stage_8[:e, :q] = X_stage_7[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_7[:e, q:2*q]
    X_stage_8[e:e*2, :q] = X_stage_7[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_7[:e, q:2*q]#skip the second half cause we dont return it
    X_stage_8[512//2 + 1,0] = X_stage_8[512//2 + 1,0].real + 0.0j#nyquist handling

    return  X_stage_8[:512//2 + 1,0]#return only the first half- the real complex


@numba.jit(numba.float64[:](numba.complex128[:],numba.complex128[:,:]),fastmath=True,nopython=True)
def unrolled_numba_irfft(n:np.ndarray, twiddlefactors: np.ndarray):
    #only works with input of size 257.
    N = 512
    #DIFFFT radix-2 simple algorithm

    # Pre-calculation of the matrix M
    M = numpy.asarray([[1.+0.0000000e+00j, 1.+0.0000000e+00j],[ 1.+0.0000000e+00j, -1.+1.2246468e-16j]],dtype=numpy.complex128) #for fft
     
    #perform the fixed extraction
    
    nFFT = 512


    X_stage_0 = np.zeros((2, 256), dtype=np.complex128)
    X_stage_1 = np.zeros((4, 128), dtype=np.complex128)
    X_stage_2 = np.zeros((8, 64), dtype=np.complex128)
    X_stage_3 = np.zeros((16, 32), dtype=np.complex128)
    X_stage_4 = np.zeros((32, 16), dtype=np.complex128)
    X_stage_5 = np.zeros((64, 8), dtype=np.complex128)
    X_stage_6 = np.zeros((128, 4), dtype=np.complex128)
    X_stage_7 = np.zeros((256, 2), dtype=np.complex128)
    X_stage_8 = np.zeros((512, 1), dtype=np.complex128)
    #exploit the symmetry
    n_full = np.hstack((n[:], np.conj(n[-2:0:-1])))
    
    #perform the initial identity transfer and unraveling.
    X_stage_0[0, :] = M[0, 0] * n_full[:256]  + M[0, 1] * n_full[256:]
    X_stage_0[1, :] = M[1, 0] * n_full[:256]  + M[1, 1] * n_full[256:]



    e = 2
    q = 128

    X_stage_1[:e, :q] = X_stage_0[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_0[:e, q:2*q]
    X_stage_1[e:e*2, :q] = X_stage_0[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_0[:e, q:2*q]

    e = 4
    q = 64
    X_stage_2[:e, :q] = X_stage_1[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_1[:e, q:2*q]
    X_stage_2[e:e*2, :q] = X_stage_1[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_1[:e, q:2*q]

    e = 8
    q = 32
    X_stage_3[:e, :q] = X_stage_2[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_2[:e, q:2*q]
    X_stage_3[e:e*2, :q] = X_stage_2[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_2[:e, q:2*q]
    
    e = 16
    q = 16
    X_stage_4[:e, :q] = X_stage_3[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_3[:e, q:2*q]
    X_stage_4[e:e*2, :q] = X_stage_3[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_3[:e, q:2*q]

    e = 32
    q = 8
    X_stage_5[:e, :q] = X_stage_4[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_4[:e, q:2*q]
    X_stage_5[e:e*2, :q] = X_stage_4[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_4[:e, q:2*q]

    e = 64
    q = 4
    X_stage_6[:e, :q] = X_stage_5[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_5[:e, q:2*q]
    X_stage_6[e:e*2, :q] = X_stage_5[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_5[:e, q:2*q]

    e = 128
    q = 2
    X_stage_7[:e, :q] = X_stage_6[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_6[:e, q:2*q]
    X_stage_7[e:e*2, :q] = X_stage_6[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_6[:e, q:2*q]

    e = 256
    q = 1
    X_stage_8[:e, :q] = X_stage_7[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_7[:e, q:2*q]
    X_stage_8[e:e*2, :q] = X_stage_7[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_7[:e, q:2*q]



    result = numpy.real(X_stage_8).flatten()
    result = result/512.0 #divide by the size of the original input

    return result

@numba.jit(numba.complex128[:,:](numba.float64[:],numba.float64[:],numba.complex128[:,:]))
def pure_numba_stft(signal:np.ndarray, window:np.ndarray,twiddle_factors: numpy.ndarray) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform (STFT) using pure Numba for a given signal.

    Parameters
    ----------
    signal : np.ndarray
        The input signal on which to compute the STFT. 
        For this implementation, it needs to always be 8192 *3 and used with a sample rate of 48k.
        If changed, other accomodations need to be made- change the padding as well as the variables.
        For perfect reconstruction, FFT size always needs to be four times the hop size,
        and the sample rate 93.75 times the fft size- 93.75 hz per frequency bin.
    window : np.ndarray
        The window function to apply during the STFT. Equal to FFT size.
    twiddle_factors : np.ndarray
        Precomputed factors used for the RFFT operation, passed to the rfft function

    Returns
    -------
    np.ndarray
        The computed STFT of the input signal.
    """

    fft_size = 512 #also the segment length
    window_size = 512
    hop_length = 128

    # Extend the input signal by mirroring at the ends
    extended_signal = np.zeros(25087, dtype=np.float64)
    extended_signal[256:-255]= signal[:]
    extended_signal[0:256] = extended_signal[257:(256*2)+1][::-1]
    extended_signal[-255:] = extended_signal[-255*2-1:-256][::-1]
    
    # Calculate parameters
    overlap_length = fft_size - hop_length
    hop_length = fft_size - overlap_length
    num_segments = (extended_signal.shape[-1] - fft_size) // hop_length + 1
    seg_len_top = int(np.ceil(fft_size / 2))
    seg_len_bottom = seg_len_top - 1 if (fft_size % 2 == 1) else seg_len_top

    # Segment the extended signal
    stft_matrix = np.zeros((fft_size, num_segments), dtype=np.float64)
    strides = (extended_signal.strides[0], hop_length * extended_signal.strides[0])

    segment_starts = np.arange(num_segments) * hop_length
    first_half = np.lib.stride_tricks.as_strided(extended_signal[seg_len_bottom:], (seg_len_top, num_segments), strides)
    second_half = np.lib.stride_tricks.as_strided(extended_signal, (seg_len_bottom, num_segments), strides)

    stft_matrix[:seg_len_top, :] = first_half
    stft_matrix[seg_len_top:, :] = second_half

    # Perform ifftshift on the window function- note that this will need to also be done in reverse on istft
    window_len = window.shape[0]
    shift_len = window_len // 2 if window_len % 2 == 0 else (window_len + 1) // 2
    window = np.concatenate((window[shift_len:], window[:shift_len]))

    # Apply windowing
    stft_matrix *= window.reshape(-1, 1)

    # Initialize the output STFT matrix
    output_stft = numpy.zeros((257,192), dtype=numpy.complex128)

    # Apply the FFT to each segment
    for seg_idx in range(stft_matrix.shape[1]):
        output_stft[:, seg_idx] =  unrolled_numba_rfft(stft_matrix[:, seg_idx], twiddle_factors)

    return output_stft
    
@numba.jit(numba.types.Tuple((numba.float64[:],numba.float64[:]))(numba.complex128[:,:],numba.float64[:],numba.float64[:],numba.complex128[:,:]),fastmath=True,nopython=True)
def numba_optimized_istft(stft_matrix: numpy.ndarray,overlap_add_buffer:numpy.ndarray,window:numpy.ndarray,twiddle_factors:numpy.ndarray):
    """
    Perform an inverse Short-Time Fourier Transform (iSTFT) on a given STFT matrix.

    Parameters
    ----------
    stft_matrix : numpy.ndarray
        The STFT matrix on which to perform the iSTFT.
    overlap_add_buffer : numpy.ndarray
        A buffer for managing overlap-add during the iSTFT.
    window : numpy.ndarray
        The window function to apply during iSTFT.
    twiddle_factors : numpy.ndarray
        Precomputed factors used for the iSTFT operation.
    
    Returns
    -------
    tuple
        The reconstructed time domain signal and the updated overlap-add buffer.
    """

    fft_size = 512
    window_size = 512
    output_size = 8192
    hop_length = 128

    # buffer to store the time domain signals obtained after applying inverse FFT
    time_domain_buffer = numpy.zeros((fft_size,64),dtype=numpy.float64)

    # inverse FFT is applied for each frame
    for frame_idx in range(stft_matrix.shape[1]):
        time_domain_buffer[:,frame_idx] =  unrolled_numba_irfft(stft_matrix[:,frame_idx],twiddle_factors)

    # Apply FFT shift
    temp = time_domain_buffer[:fft_size//2,:].copy()
    time_domain_buffer[:fft_size//2,:] = time_domain_buffer[fft_size//2:,:]
    time_domain_buffer[fft_size//2:,:] = temp[:]

    # Windowing is applied for each frame
    for frame_idx in range(time_domain_buffer.shape[1]):
        time_domain_buffer[:,frame_idx] =  time_domain_buffer[:,frame_idx] * window

    # Initialize the output buffer
    output_buffer = numpy.zeros(output_size, dtype=numpy.float64)

    # Overlap-add method is applied for reconstructing the time-domain signal
    time_idx = 0
    for frame_idx in range(time_domain_buffer.shape[1]):
        # Overlap-add
        output_buffer[time_idx : time_idx + hop_length] = overlap_add_buffer[:128] + time_domain_buffer[0 : hop_length, frame_idx]

        # Shift the overlap-add buffer
        overlap_add_buffer[: -hop_length] = overlap_add_buffer[hop_length:]  # shift out left
        overlap_add_buffer[-hop_length :] = 0.0

        # Add the overlap region for next frame
        overlap_add_buffer[:] += time_domain_buffer[-384:, frame_idx] #n_FFT - hop_length
        time_idx += hop_length
    
    return output_buffer, overlap_add_buffer
