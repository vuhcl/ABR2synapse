import numpy as np
from scipy.interpolate import CubicSpline


def find_crossings(wave, zero=False):
    if not zero:
        baseline = (wave[-1] + wave[0]) / 2
    else:
        baseline = 0
    x_axis = range(len(wave))
    zero_crossings = np.where(np.diff(np.sign(wave - baseline)))[0]
    return [x_axis[indice] for indice in zero_crossings]


def find_waveI(wave, peaks, troughs):
    zero_crossings = find_crossings(wave[: peaks[0]], True)
    if zero_crossings:
        start = zero_crossings[-1]
    else:
        start = 0
    mid = peaks[0] + find_crossings(wave[peaks[0] : troughs[0] + 1])[0]
    if peaks[0 + 1] > troughs[0]:
        end = troughs[0] + find_crossings(wave[troughs[0] : peaks[0 + 1] + 1])[0]
    else:
        end = troughs[0] + find_crossings(wave[troughs[0] : troughs[0] + 10])[0]
    return start, mid, end


def slope(x1, x2, y1, y2):
    if x2 - x1 == 0:
        return 0
    return (y2 - y1) / (x2 - x1)


def calculate_slope(wave, peaks, troughs):
    x = np.linspace(0, 17, len(wave))
    start, mid, end = find_waveI(wave, peaks, troughs)
    return slope(x[peaks[0]], x[mid], wave[peaks[0]], wave[mid])


def calculate_variance(wave, peaks, troughs):
    start, mid, end = find_waveI(wave, peaks, troughs)
    waveI = wave[start : end + 1]
    return np.sum((waveI - np.mean(waveI)) ** 2)


def calculate_distance(wave, peaks, troughs):
    x_axis = np.linspace(0, 17, len(wave))
    return x_axis[troughs[0]] - x_axis[peaks[0]]


def calculate_curvature(wave, point):
    x_axis = np.linspace(0, 17, len(wave))
    idx = range(point - 10, point + 11)
    x = x_axis[idx]
    y = wave[idx]
    cs = CubicSpline(x, y)
    values = x_axis[point - 3 : point + 4 : 3]
    return np.abs(cs(values, 2)) / (1 + cs(values, 1) ** 2) ** (3 / 2)
