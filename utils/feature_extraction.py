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


def slope_from_amplitude_distance(amplitude: float, distance_ms: float) -> float:
    """
    Row-level Wave-I slope used in Liberman loading: ``-(amplitude / distance_ms)`` (µV/ms).

    ``amplitude`` and ``distance_ms`` are the same quantities as the stored ``Amplitude``
    and ``Distance`` columns (distance from ``calculate_distance``, i.e. latency span in ms).
    """
    if distance_ms is None or not np.isfinite(distance_ms) or float(distance_ms) <= 0:
        return float("nan")
    return -float(amplitude) / float(distance_ms)


def calculate_variance(wave, peaks, troughs):
    start, mid, end = find_waveI(wave, peaks, troughs)
    waveI = wave[start : end + 1]
    return np.sum((waveI - np.mean(waveI)) ** 2)


def calculate_distance(wave, peaks, troughs):
    x_axis = np.linspace(0, 17, len(wave))
    return x_axis[troughs[0]] - x_axis[peaks[0]]


def calculate_curvature(wave, point):
    """Graph curvature κ at three times around `point` (early / central / late on the trace)."""
    x_axis = np.linspace(0, 17, len(wave))
    idx = range(point - 10, point + 11)
    x = x_axis[idx]
    y = wave[idx]
    cs = CubicSpline(x, y)
    values = x_axis[point - 3 : point + 4 : 3]
    return np.abs(cs(values, 2)) / (1 + cs(values, 1) ** 2) ** (3 / 2)


def calculate_p1_latency_ms(wave, peak_index):
    """Map peak sample index to time (ms) on the 0–17 ms axis (matches Brad convention)."""
    wave = np.asarray(wave)
    if wave.size == 0:
        return np.nan
    t_ms = np.linspace(0, 17, len(wave))
    pi = int(peak_index)
    if pi < 0 or pi >= len(t_ms):
        return np.nan
    return float(t_ms[pi])


def amplitude_vs_spl_slopes(levels_db, amplitudes):
    """
    Linear slope of Wave I amplitude vs SPL (dB): all levels (Slope_all) and high-SPL band
    (Slope_high4: SPL >= 65 dB if >= 2 points, else four highest SPLs).
    """
    levels_db = np.asarray(levels_db, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    mask = np.isfinite(levels_db) & np.isfinite(amplitudes)
    levels_db = levels_db[mask]
    amplitudes = amplitudes[mask]
    if levels_db.size < 2:
        return np.nan, np.nan
    order = np.argsort(levels_db)
    spl = levels_db[order]
    amp = amplitudes[order]
    slope_all = float(np.polyfit(spl, amp, 1)[0])
    high = spl >= 65
    if np.sum(high) >= 2:
        slope_h4 = float(np.polyfit(spl[high], amp[high], 1)[0])
    else:
        tail = np.argsort(spl)[-min(4, len(spl)) :]
        if len(np.unique(spl[tail])) < 2:
            slope_h4 = np.nan
        else:
            slope_h4 = float(np.polyfit(spl[tail], amp[tail], 1)[0])
    return slope_all, slope_h4
