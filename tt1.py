import numpy as np
import pywt
import matplotlib.pyplot as plt

ecg = np.loadtxt('ecg_segment2.txt')
fs = 200
w = pywt.Wavelet('haar')
ca = []
cd = []
layers = int(7)
mode = pywt.Modes.smooth
a = ecg
for i in range(layers):
    print(i)
    (a, d) = pywt.dwt(a, w, mode)
    print('length of approcimate coeff ', len(a))
    print('length of detail coeff ', len(d))
    print('\n')
    ca.append(a)
    cd.append(d)


rec_a = []
rec_d = []
for i, coeff in enumerate(ca):
    coeff_list = [coeff, None] + [None] * i
    rec_a.append(pywt.waverec(coeff_list, w))

for i, coeff in enumerate(cd):
    coeff_list = [None, coeff] + [None] * i
    rec_d.append(pywt.waverec(coeff_list, w))



D1 = rec_d[0]
D2 = rec_d[1]
D3 = rec_d[2]
print('length of D1 {0}, length of D2 {1}, length of D3 {2}'.format(len(D1),len(D2),len(D3)))
# HF= D1+D2+D3
HF = D1 + D2


## compute the zero crossing

size_block = int(0.05*fs)
shift_step = 1
size_epoch = int(10*fs)
st = 0
stp = st+size_epoch
thd_zc = 2
while stp <= len(HF):
    epoch = HF[st:stp]
    st_block = 0
    stp_block = st_block+size_block
    ZCE = []
    while stp_block <= len(epoch):
        block = epoch[st_block:stp_block]
        zc = 0
        if np.mean(abs(block))>thd_zc:
            block[block > 0] = 1
            block[block < 0] = -1
            cross_zero = np.abs(np.diff(block))
            zc = len(np.argwhere(cross_zero>0))/size_block
        ZCE.append(zc)
        st_block = st+1
        stp_block = st_block+size_block

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(epoch)
    ax[1].plot(ZCE)
    ax[1].plot()
    plt.show()
    print()















































D4 = rec_d[3][0:len(ecg)]
D5 = rec_d[4][0:len(ecg)]
D6 = rec_d[5][0:len(ecg)]
print('length of D4 {0}, length of D5 {1}, length of D6 {2}'.format(len(D4),len(D5),len(D6)))
ECG= D4+D5+D6


fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
ax[0].plot(ecg)
ax[1].plot(D1,label='D1')
ax[2].plot(D2,label='D2')
# ax[3].plot(D3,label='D3')
ax[3].plot(ECG,label='D3')
ax[4].plot(HF,label= 'HF')
# ax[2].plot()

plt.show()



#
#
#
#
#
#
#
#
#
#
#
# fig = plt.figure()
# ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
# # ax_main.set_title(title)
# ax_main.plot(ecg)
# ax_main.set_xlim(0, len(ecg) - 1)
#
# for i, y in enumerate(rec_a):
#     ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
#     ax.plot(y, 'r')
#     ax.set_xlim(0, len(y) - 1)
#     ax.set_ylabel("A%d" % (i + 1))
#
# for i, y in enumerate(rec_d):
#     ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
#     ax.plot(y, 'g')
#     ax.set_xlim(0, len(y) - 1)
#     ax.set_ylabel("D%d" % (i + 1))
# plt.show()
#
# a8 = rec_a[6]
# half = int((abs(len(a8)-len(ecg)))/2)
# a8 = a8[half:len(a8)-half]
#
# fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
# ax[0].plot(ecg)
# ax[0].plot(rec_a[6], 'r')
# ax[0].plot(a8, 'g')
#
# ax[1].plot(ecg-rec_a[6][:len(ecg)])
# ax[2].plot(ecg - a8)
#
# plt.show()
#
#
