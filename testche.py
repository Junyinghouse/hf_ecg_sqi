
import os
import numpy as np
import ParseCHE

def find_che(dir):
    data = None
    if os.path.isdir(dir):
        ll = os.listdir(dir)
        for line in ll:
            filepath = os.path.join(dir, line)
            if os.path.isfile(filepath):
                if (filepath.endswith(".CHE")):
                    parseche = ParseCHE.ParseCHE()
                    data = parseche.parse(filepath)
            else:
                find_che(filepath)
    return data

def smooth_l(signal, size):
    smooth = np.zeros(len(signal))

    for i in np.arange(len(signal)):
        if i == 0:
            smooth[i] = signal[i]
        elif i < np.divide((size - 1), 2):
            smooth[i] = np.mean(signal[:i * 2 + 1])
        elif i > len(signal) - np.divide((size - 1), 2) - 1:
            smooth[i] = np.mean(signal[i - (len(signal) - 1 - i):len(signal)])
        else:
            start = int(i - np.divide((size - 1), 2))
            # start.astype(np.int16)
            end = int(i + 1 + np.divide((size - 1), 2))
            # end.astype(np.int16)
            smooth[i] = np.mean(signal[start:end])
    return smooth

def static(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    x = abs(np.divide((a - 512), 128))
    y = np.divide((b - 512), 128)
    z = np.divide((c - 512), 128)

    svm = (x * x + y * y + z * z) ** 0.5
    svm_sm = smooth_l(svm, size=3)
    d_svm = np.sum(np.abs(np.diff(svm_sm)))
    x1 = np.mean(smooth_l(x, 3))
    y1 = np.mean(smooth_l(y, 3))
    z1 = np.mean(smooth_l(z, 3))
    position = 9
    if d_svm >= 1 and d_svm < 6:
        position = 7
    elif d_svm >= 6:
        position = 8
    elif d_svm >= 0.35 and d_svm < 1:
        position = 6
    elif d_svm < 0.35:
        if x1 >= 0.8:
            position = 1
        elif z1 >= 0.5:
            position = 2
        elif z1 < -0.5:
            position = 3
        elif y1 >= 0.5:
            position = 4
        elif y1 < -0.5:
            position = 5
    return position


if __name__ == '__main__':

    fs_ecg = 200
    fs_xyz = 25
    path_dir = "D:\\hs_work_2017\\hs_data\\sleep_data\\fang_junying"
    data = find_che(path_dir)
    ecg = data['ecgList']
    x_mv = data['xList']
    y_mv = data['yList']
    z_mv = data['zList']
    print(data.keys())
    # position = static(x_mv, y_mv, z_mv)

    win = 5  # the width of epoch
    e = []
    st1,stp1 = 0, win*fs_ecg - 1
    while stp1 <= len(ecg):
        ep1 = ecg[st1:stp1]
        st1 = stp1+1
        stp1 = st1+win*fs_ecg-1
        e.append(ep1)

    epoch_ecg = np.asarray(e)
    print(epoch_ecg.shape)















