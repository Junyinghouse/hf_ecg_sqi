import wavelet_decomp_reconstr as AD
import numpy as np

ecg = np.loadtxt('ecg_segment2.txt')
A_D = AD.wavelet_reconstruction(ecg)
