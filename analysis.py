from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pyts.decomposition import SingularSpectrumAnalysis
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import butter, find_peaks, sosfiltfilt

from config import (
    FC_HIGH,
    FC_LOW,
    MAX_WORKERS,
    SAMPLING_FREQUENCY,
    SSA_CHUNKSIZE,
    SSA_COMPONENTS,
    SSA_WINDOW,
)

def read_signal(path: Path) -> np.ndarray:
    signal = np.fromfile(path, dtype="<i2").astype(float)
    if signal.size < 2:
        raise ValueError("file too small")

    nyquist = 0.5 * SAMPLING_FREQUENCY 
    sos = butter(
        N=2,
        Wn=[FC_HIGH / nyquist, FC_LOW / nyquist],
        btype="band",
        output="sos",
    )
    signal = sosfiltfilt(sos, signal)

    if not np.all(np.isfinite(signal)):
        raise ValueError("nan/inf")
    if np.allclose(signal, signal[0]):
        raise ValueError("flat signal")

    return signal


def fft_spectrum(
    signal: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    mags = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1 / SAMPLING_FREQUENCY)
    mask = (freqs >= FC_HIGH) & (freqs <= FC_LOW)

    peak = float(freqs[mask][np.argmax(mags[mask])])
    return peak, freqs, mags


def _fit_sinusoid_frequency(signal: np.ndarray) -> tuple[float, float, float]:
    time_seconds = np.arange(signal.size, dtype=float) / SAMPLING_FREQUENCY

    smoothed_signal = gaussian_filter1d(signal, sigma=2)
    min_peak_distance = int(0.5 * SAMPLING_FREQUENCY)
    peaks, _ = find_peaks(smoothed_signal, distance=min_peak_distance)

    if len(peaks) >= 2:
        period = np.median(np.diff(peaks) / SAMPLING_FREQUENCY)
        f0 = 1.0 / period if period > 0 else 1.0
    else:
        f0 = 1.0
    f0 = float(np.clip(f0, FC_HIGH, FC_LOW))

    def residuals(p):
        ''' 
        A: Амплитуда
        w: Угловая частота
        phi: Фаза
        c: Смещение
        '''
        A, w, phi, c = p
        return A * np.sin(w * time_seconds + phi) + c - signal

    result = least_squares(
        residuals,
        x0=[
            np.std(signal) * np.sqrt(2),
            2 * np.pi * f0,
            0.0,
            np.mean(signal),
        ],
        bounds=(
            [0, 2 * np.pi * FC_HIGH, -np.pi, -np.inf],
            [np.inf, 2 * np.pi * FC_LOW, np.pi, np.inf],
        ),
    )

    A, w, _, _ = result.x
    frequency = w / (2 * np.pi)
    mse = float(np.mean(result.fun**2))
    return float(frequency), float(A), mse


def dominant_ssa_frequency(
    signal: np.ndarray
) -> float:
    if signal.size < 50:
        raise ValueError("signal too short for SSA")

    ssa = SingularSpectrumAnalysis(
        window_size=SSA_WINDOW,
        groups=[[i] for i in range(SSA_COMPONENTS)],
        chunksize=SSA_CHUNKSIZE,
    )
    decomposition = ssa.fit_transform(signal.reshape(1, -1))
    comps = decomposition[0] if decomposition.ndim == 3 else decomposition

    best_f = FC_HIGH
    best_score = np.inf

    for i in range(0, len(comps) - 1, 2):
        paired = comps[i] + comps[i + 1]
        frequency, _, mse = _fit_sinusoid_frequency(paired)
        score = mse / (np.var(paired) + 1e-3)

        if score < best_score:
            best_score = score
            best_f = frequency

    return best_f


def _analyze_one(
    path: Path,
) -> tuple[Path, dict | None, str | None]:
    try:
        signal = read_signal(path)
        fft_hz, freqs, mags = fft_spectrum(signal)
        ssa_hz = dominant_ssa_frequency(signal)
        return path, {"fft_hz": fft_hz, "ssa_hz": ssa_hz, "freqs": freqs, "mags": mags}, None
    except Exception as e:
        return path, None, str(e)


def analyze_files(
    files: list[Path]
) -> list[dict]:
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, (path, data, error) in enumerate(
            pool.map(_analyze_one, files),
        ):
            if error:
                print(f"[{i}/{len(files)}] {path.stem} — skip: {error}")
                continue
            print(
                f"[{i}/{len(files)}] {path.stem} — "
                f"{data['fft_hz'] * 60:.2f} / {data['ssa_hz'] * 60:.2f} BPM"
            )
            results.append({"path": path, **data})
    return results
