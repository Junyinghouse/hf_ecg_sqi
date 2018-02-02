import wavelet_decomp_reconstr as dec_rec_wavelet
import numpy as np
import matplotlib.pyplot as plt
import nolds as nd
from scipy import stats,signal

def zero_cross(x):

    sign_x = np.sign(x)
    cross_zero = np.abs(np.diff(sign_x))
    nb_zc = len(np.argwhere(cross_zero > 0)) / len(x)
    return nb_zc

def smooth_filter(mylist, N):
    y = []
    st = [i for i in range(1, N) if i % 2 == 1]
    sp = [i for i in range(1, N) if i % 2 == 1]  ## be careful reverse functino to itself
    sp.reverse()
    mn = [N for i in range(len(mylist) - 2 * len(st))]
    win = st + mn + sp
    for i in range(len(mylist)):
        if i == 0 or i == len(mylist) - 1:
            y.append(mylist[i])
        else:
            # y.append(np.mean(mylist[int(i - (win[i] - 1) / 2):int(i + (win[i] - 1) / 2)]))
            y.append(np.median(mylist[int(i - (win[i] - 1) / 2):int(i + (win[i] - 1) / 2)]))
    return y

def width_gate(gate_signal):
    x = np.asarray(gate_signal)
    idx = np.argwhere(x>0)
    widthGate_st,widthGate_stp = [],[]
    if len(idx):
        start_current = idx[0]
        widthGate_st.append(start_current)
        for i in range(1,len(idx)):
            if abs(idx[i] - idx[i-1]) > 1:
                stop_current = idx[i-1]
                widthGate_stp.append(stop_current)
                widthGate_st.append(idx[i])
        widthGate_stp.append(idx[-1])

    widthGata = [widthGate_st, widthGate_stp]

    return widthGata

def dec_all_peaks(signal):
    '''
    波峰检测：所有波峰
    :param ss: 输入信号原始波形
    :return: 所有波峰的索引位置
    '''
    s2 = []
    diff_ss = signal[:len(signal) - 1] - signal[1:len(signal)]
    for sig in diff_ss:
        s1 = np.sign(sig)
        s2.append(s1)
    s2 = np.array(s2)
    diff_s2 = s2[:len(s2) - 1] - s2[1:len(s2)]
    xx2 = range(len(diff_s2))
    maxtab2 = []
    for i2 in range(len(xx2)):
        if diff_s2[i2] < 0:
            maxpos2 = xx2[i2] + 1
            maxtab2.append(maxpos2)
    return maxtab2




if __name__ == '__main__':

    ecg_x = np.loadtxt('ecg_segment3.txt')
    fs = 200
    A_D = dec_rec_wavelet.wavelet_reconstruction(ecg_x)
    HF = A_D[0] + A_D[1]

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(ecg_x,label='raw ecg')  # plot raw ECG signal
    ax[0].legend(loc='lower right')

    ax[1].plot(HF, label='HF')
    ax[1].legend(loc='lower right')
    plt.show()




    size_epoch = int(10*fs)
    ovlp = 0.5
    step = int((1-ovlp)*fs)
    st = 0
    stp = st + size_epoch

    size_block = int(0.1 * fs) # block窗口宽度， 经验值 (ZCE)
    shift_step = 1

    size_block2 = int(0.2*fs)  # block窗口宽度（ACF）
    shift_step2 = int(0.2*size_block2) # shift 窗口长度的20%


    thd_zc = 7  # QRS幅值经验值，和个体，电极贴位置有关， 差异性很大
    thd_1 = 0.5 # 噪声幅值经验值
    thd_2 = int(0.3*fs) # 和 RR间期波动有关
    thd_kur1 = 2 # kurtosis of the ECG（kSQI）:clean ECG sharp peak of the distribution >5 .  白噪声round 3
    thd_kur2 = 4

    t = []

    while stp <= len(HF): # 一段ecg > 10秒
        t.append(st)
        epoch_ecg = ecg_x[st:stp]
        epoch_HF = HF[st:stp]

        H3 = 0
        type_noise = []

        #### 10秒窗口下计算HF，过零次数ZCE曲线 （vector）
        st_block = 0
        stp_block = st_block+size_block
        ZCE = []
        while stp_block <= len(epoch_HF):
            block = epoch_HF[st_block:stp_block]
            nb_zc = 0
            zk = np.mean(abs(block))
            if zk > thd_zc:
                nb_zc = zero_cross(block)
            ZCE.append(nb_zc)
            st_block = st_block+shift_step
            stp_block = st_block+size_block

        ### 是否存在高频噪声H3
        smooth_zce = smooth_filter(ZCE, 5)
        smo_zce1 = np.asarray(smooth_zce)
        sign_zce1 = np.sign(smo_zce1)
        widthGate = width_gate(sign_zce1)
        W_zec = np.asarray([widthGate[1][i] - widthGate[0][i] for i in range(len(widthGate[0]))])
        idx = np.argwhere(W_zec >= thd_2)

        ### 10秒窗口下计算D1，kurtosis值（value）
        if np.max(smo_zce1) and len(idx):
            H3 = 1
            kur_D1= stats.kurtosis(A_D[0][st:stp],
                                        fisher=False)  # kurtosis of the ECG（kSQI）:clean ECG sharp peak of the distribution >5 . noise<=5
            if kur_D1 > thd_kur1  and kur_D1 < thd_kur2:
                type_noise = 'AWGN'

            else:
                ### 10秒窗口下计算自相关 ACF，vector
                st_block2 = 0
                stp_block2 = st_block2 + size_block2
                ACF = []
                while stp_block2 <= len(epoch_HF):
                        block2 = epoch_HF[st_block2:stp_block2]
                        y = block2  - np.mean(block2 )
                        norm = np.sum(y ** 2)
                        correlated = np.correlate(y, y, mode='full') / norm
                        m = int((correlated.shape[0] + 1) / 2)
                        t_lags = np.linspace(0, m - 1, m) / fs
                        acor = correlated[m - 1:]
                        ACF.append(dec_all_peaks(acor))
                






        st = st + step
        stp = st+size_epoch
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        t1 = np.linspace(0, size_epoch - 1, size_epoch) / fs
        t2 = np.linspace(0, size_epoch - 1, size_epoch) / fs
        t3 = np.linspace(0, len(ZCE) - 1, len(ZCE)) / (1 / (shift_step / fs))
        ax[0].plot(t1, epoch_ecg)
        ax[1].plot(t2, epoch_HF)
        ax[2].plot(t3, ZCE)
        plt.show()



    #
    # compute the zero crossing

    size_block = int(0.1 * fs)
    shift_step = 1
    thd_zc = 7
    st_block = 0
    stp_block = st_block + size_block
    ZCE = []
    while stp_block <= len(ecg_x):
        block = HF[st_block:stp_block]
        nb_zc = 0
        zk = np.mean(abs(block))
        if zk > thd_zc:
            nb_zc = zero_cross(block)
        ZCE.append(nb_zc)
        st_block = st_block + shift_step
        stp_block = st_block + size_block

    smooth_zce = smooth_filter(ZCE, 5)
    smo_zce1 = np.asarray(smooth_zce)
    sign_zce1 = np.sign(smo_zce1)
    widthGate = width_gate(sign_zce1)
    W_zec = [widthGate[1][i] - widthGate[0][i] for i in range(len(widthGate[0]))]


    sign_zce = np.sign(ZCE)
    smo_zce1[sign_zce==0] = 0

    # mask_diff= np.argwhere((np.sign(ZCE) > 0)^(np.sign(smo_zce1)>0) == True )







    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
    t1 = np.linspace(0, len(ecg_x) - 1, len(ecg_x)) / fs
    t2 = np.linspace(0, len(HF) - 1, len(HF)) / fs
    t3 = np.linspace(0, len(ZCE) - 1, len(ZCE)) / (1 / (shift_step / fs))
    ax[0].plot(ecg_x)
    ax[1].plot(HF)
    ax[2].plot(ZCE)
    ax[3].plot(smooth_zce)
    ax[4].plot(sign_zce1)
    ax[4].scatter(widthGate[0],sign_zce1[widthGate[0]],c='r')
    ax[4].scatter(widthGate[1],sign_zce1[widthGate[1]],c='g')
    # ax[0].plot(t1, ecg_x)
    # ax[1].plot(t2, HF)
    # ax[2].plot(t3, ZCE)
    # ax[3].plot(t3, smooth_zce)
    # ax[4].plot(t3, sign_zce1)
    plt.show()




##






































# SE =[]
# t = []
# while stp <= len(HF):
#     t.append(st)
#     epoch_ecg = ecg[st:stp]
#     epoch = HF[st:stp]
#     # se = nd.sampen(epoch, emb_dim=2, tolerance=0.15 * np.std(epoch))  ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     # plt.figure()
#     # plt.plot(epoch)
#     # plt.show()
#     st_block = 0
#     stp_block = st_block+size_block
#     ZCE = []
#     # # SE =[]
#     while stp_block <= len(epoch):
#         block = epoch[st_block:stp_block]
#         # se = nd.sampen(block, emb_dim=2, tolerance=0.15 * np.std(block))  ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         SE.append(se)
#         # plt.figure()
#         # plt.plot(block)
#         # plt.show()
#         zc = 0
#         zk = np.mean(abs(block))
#         # print(zk)
#         if zk > thd_zc:
#             # copy_block = block
#             # copy_block[copy_block > 0] = 1
#             # copy_block[copy_block < 0] = -1
#             sign_block= np.sign(block)
#             cross_zero = np.abs(np.diff(sign_block))
#             zc = len(np.argwhere(cross_zero>0))/size_block
#         ZCE.append(zc)
#         st_block = st_block+shift_step
#         stp_block = st_block+size_block
#
#
#     st = st + step
#     stp = st+size_epoch
#     # SE.append(se)
#     fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
#     ax[0].plot(epoch_ecg)
#     ax[1].plot(epoch)
#     ax[2].plot(ZCE)
#     plt.show()
#












# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# ax[0].plot(range(len(ecg)),ecg)
# ax[1].plot(t,SE)
# plt.show()
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # ax[0].plot(epoch_ecg)
    # ax[1].plot(epoch)
    # ax[2].plot(ZCE)
    # ax[3].plot(SE)
    # plt.show()
    # print()
