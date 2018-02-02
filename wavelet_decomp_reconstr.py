import pywt
import numpy as np

def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))

def wavelet_reconstruction(ecg, decomp_level=7, wavelet = 'haar'):
    '''

    :param ecg: 原始ecg信号
    :param decomp_level: 分解的层数
    :param wavelet: 小波基，默认haar
    :return: 一共由decomp_level+1个list组成，按照从前往后依次为D1，D2，...D7,A7。
    '''
    X = ecg
    level = decomp_level
    coeffs = pywt.wavedec(X, wavelet, level=level)
    results = []
    for i in range(level):  # 所有的detail
        results.append(wrcoef(X, 'd', coeffs, wavelet, i+1))
    results.append(wrcoef(X, 'a', coeffs, wavelet, level))

    return results
