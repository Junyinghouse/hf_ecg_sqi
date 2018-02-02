import wavelet_decomp_reconstr as dec_rec_wavelet
import numpy as np
import matplotlib.pyplot as plt
import nolds as nd

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

    if len():
        pass

    pass



if __name__ == '__main__':

    ecg_x = np.loadtxt('ecg_segment3.txt')
    fs = 200
    A_D = dec_rec_wavelet.wavelet_reconstruction(ecg_x)
    HF = A_D[0] + A_D[1]
    #
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # ax[0].plot(ecg,label='raw ecg')  # plot raw ECG signal
    # ax[0].legend(loc='lower right')
    #
    # ax[1].plot(HF, label='HF')
    # ax[1].legend(loc='lower right')
    # plt.show()

    # size_epoch = int(10*fs)
    # size_block = int(0.1 * fs) # block窗口宽度， 经验值
    # shift_step = 1
    # ovlp = 0.5
    # step = int((1-ovlp)*fs)
    # st = 0
    # stp = st + size_epoch
    # thd_zc = 7  # QRS幅值经验值，和个体，电极贴位置有关， 差异性很大
    # thd_1 = 0.5 # 噪声幅值经验值
    # thd_2 = int(0.3*fs) # 和 RR间期波动有关
    # t = []
    #
    # while stp <= len(HF): # 一段ecg > 10秒
    #
    #     t.append(st)
    #     epoch_ecg = ecg_x[st:stp]
    #     epoch_HF = HF[st:stp]
    #     st_block = 0
    #     stp_block = st_block+size_block
    #     ZCE = []
    #     #### 10秒窗口下计算HF，过零次数ZCE曲线 （vector）
    #     while stp_block <= len(epoch_HF):
    #         block = epoch_HF[st_block:stp_block]
    #         nb_zc = 0
    #         zk = np.mean(abs(block))
    #         if zk > thd_zc:
    #             nb_zc = zero_cross(block)
    #         ZCE.append(nb_zc)
    #         st_block = st_block+shift_step
    #         stp_block = st_block+size_block
    #
    #     ### 是否存在高频噪声H3
    #
    #
    #
    #     ### 10秒窗口下计算D1，kurtosis值（value）
    #
    #
    #
    #
    #
    #     st = st + step
    #     stp = st+size_epoch
    #     fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    #     t1 = np.linspace(0, size_epoch - 1, size_epoch) / fs
    #     t2 = np.linspace(0, size_epoch - 1, size_epoch) / fs
    #     t3 = np.linspace(0, len(ZCE) - 1, len(ZCE)) / (1 / (shift_step / fs))
    #     ax[0].plot(t1, epoch_ecg)
    #     ax[1].plot(t2, epoch_HF)
    #     ax[2].plot(t3, ZCE)
    #     plt.show()



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





    sign_zce = np.sign(ZCE)
    smo_zce1[sign_zce==0] = 0

    # mask_diff= np.argwhere((np.sign(ZCE) > 0)^(np.sign(smo_zce1)>0) == True )







    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
    t1 = np.linspace(0, len(ecg_x) - 1, len(ecg_x)) / fs
    t2 = np.linspace(0, len(HF) - 1, len(HF)) / fs
    t3 = np.linspace(0, len(ZCE) - 1, len(ZCE)) / (1 / (shift_step / fs))
    ax[0].plot(t1, ecg_x)
    ax[1].plot(t2, HF)
    ax[2].plot(t3, ZCE)
    ax[3].plot(t3, smooth_zce)
    ax[4].plot(t3, sign_zce1)
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
