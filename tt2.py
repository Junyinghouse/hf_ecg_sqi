import numpy as np
import pywt
import matplotlib.pyplot as plt

ecg = np.loadtxt('ecg_segment4.txt')
# plt.plot(ecg)
# plt.show()
w = pywt.Wavelet('db3')
ca = []
cd = []
layers = int(7)
mode = pywt.Modes.smooth
a = ecg
for i in range(layers):
    (a, d) = pywt.dwt(a, w, mode)
    ca.append(a)
    cd.append(d)

rec_a = []
rec_d = []
coeff_list_new = []
# for i, coeff in enumerate(cd):
#     coeff_list = [coeff, None] + [None] * i
#     coeff_list_new.append()
#     rec_a.append(pywt.waverec(coeff_list, w))

for i in range(3, 4, 5):
    coeff_list = cd[i]
    coeff_list_new.append(coeff_list)
re_ecg = pywt.waverec(coeff_list_new, w)


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(ecg)
ax[1].plot(re_ecg)
plt.show()


