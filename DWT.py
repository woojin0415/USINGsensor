import pywt
import numpy as np
def dwt(data, iter):
    if iter == 0:
        return data
    ca = data
    for k in range(iter):
        (ca, cd) = pywt.dwt(ca, "haar")
    cat = pywt.threshold(ca, np.std(ca), mode="soft")
    cdt = pywt.threshold(cd, np.std(cd), mode="soft")
    tx = pywt.idwt(cat, cdt, "haar")
    return ca.tolist()

def dwt_denoise(data, iter=0):
    (ca, cd) = pywt.dwt(data, "haar")
    for k in range(iter):
        (ca, cd) = pywt.dwt(ca, "haar")
    cat = pywt.threshold(ca, np.std(ca), mode="soft")
    cdt = pywt.threshold(cd, np.std(cd), mode="soft")
    tx = pywt.idwt(cat, cdt, "haar")
    return tx