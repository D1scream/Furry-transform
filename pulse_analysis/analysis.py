from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Препроцессинг ECG
TIME_START = 10.0
TIME_END = 110.0
FC_HIGH = 0.5
FC_LOW = 5.0
DEFAULT_SIGNAL_COLUMN = 2
SSA_COMPONENTS = 4


def _validate_signal(signal: np.ndarray, sampling_frequency: float) -> None:
    """Проверяет, что сигнал пригоден для спектрального анализа."""

    if sampling_frequency <= 0:
        raise ValueError("Частота дискретизации должна быть положительной.")
    if signal.size < 2:
        raise ValueError("Сигнал должен содержать минимум два отсчета.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Сигнал содержит NaN или бесконечные значения.")
    if np.allclose(signal, signal[0]):
        raise ValueError("Сигнал вырожден: все отсчеты одинаковы.")


def _preprocess_ecg(signal: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """Полосовой фильтр ECG в диапазоне 0.5–5 Гц."""

    nyquist = 0.5 * sampling_frequency
    sos = butter(N=2, Wn=[FC_HIGH / nyquist, FC_LOW / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def read_signal(
    path: Path,
    signal_column: int = DEFAULT_SIGNAL_COLUMN,
) -> tuple[np.ndarray, float]:
    """Читает ECG из CSV, обрезает по времени и препроцессирует сигнал."""

    try:
        data = np.loadtxt(path, delimiter=",")
    except Exception as error:
        raise ValueError(f"Не удалось прочитать CSV-файл {path}: {error}") from error

    if data.ndim != 2 or data.shape[1] <= signal_column:
        raise ValueError(f"Файл {path} не содержит колонку сигнала с индексом {signal_column}.")

    time = np.asarray(data[:, 0], dtype=float)
    signal = np.asarray(data[:, signal_column], dtype=float)
    mask = (time >= TIME_START) & (time <= TIME_END)
    time = time[mask]
    signal = signal[mask]

    if time.size < 2:
        raise ValueError(f"Файл {path} содержит слишком мало временных отсчетов.")
    time_steps = np.diff(time)
    median_step = float(np.median(time_steps))

    sampling_frequency = 1.0 / median_step
    signal = _preprocess_ecg(signal, sampling_frequency)
    _validate_signal(signal, sampling_frequency)
    return signal, sampling_frequency


def dominant_frequency_with_magnitude(
    signal: np.ndarray,
    sampling_frequency: float,
    min_frequency: float = FC_HIGH,
    max_frequency: float = FC_LOW,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Возвращает частоту пика, амплитуду и спектр |FFT|."""

    _validate_signal(signal, sampling_frequency)
    centered_signal = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered_signal)
    frequencies = np.fft.rfftfreq(centered_signal.size, d=1.0 / sampling_frequency)
    magnitudes = np.abs(spectrum)

    start_index = int(np.searchsorted(frequencies, min_frequency))
    end_index = int(np.searchsorted(frequencies, max_frequency, side="right"))
    if start_index >= end_index:
        raise ValueError(
            f"Диапазон частот [{min_frequency}, {max_frequency}] Гц вне спектра сигнала."
        )

    band = magnitudes[start_index:end_index]
    peak_index = start_index + int(np.argmax(band))

    return (
        float(frequencies[peak_index]),
        float(magnitudes[peak_index]),
        frequencies,
        magnitudes,
    )


def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    """Восстанавливает одномерный ряд из матрицы компоненты SSA."""

    rows, columns = matrix.shape
    diagonal_indices = np.add.outer(np.arange(rows), np.arange(columns)).ravel()
    restored = np.bincount(
        diagonal_indices,
        weights=matrix.ravel(),
        minlength=rows + columns - 1,
    )
    counts = np.bincount(diagonal_indices, minlength=rows + columns - 1)
    return restored / counts


def dominant_ssa_frequency(
    signal: np.ndarray,
    sampling_frequency: float,
    window: int,
    components_count: int = SSA_COMPONENTS,
    min_frequency: float = FC_HIGH,
) -> float:
    """Ищет основную частоту среди первых компонент SSA."""

    _validate_signal(signal, sampling_frequency)
    if not 2 <= window <= signal.size:
        raise ValueError(
            f"Окно SSA должно быть в диапазоне [2, {signal.size}], получено {window}."
        )

    centered_signal = signal - np.mean(signal)
    trajectory = np.lib.stride_tricks.sliding_window_view(centered_signal, window).T
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        trajectory,
        full_matrices=False,
    )

    best_frequency = 0.0
    best_peak_magnitude = -np.inf
    components_count = min(components_count, singular_values.size)

    for index in range(components_count):
        elementary_matrix = singular_values[index] * np.outer(
            left_vectors[:, index],
            right_vectors_t[index, :],
        )
        component = diagonal_averaging(elementary_matrix)
        frequency, peak_magnitude, _, _ = dominant_frequency_with_magnitude(
            component,
            sampling_frequency,
            min_frequency,
        )

        if peak_magnitude > best_peak_magnitude:
            best_peak_magnitude = peak_magnitude
            best_frequency = frequency

    return float(best_frequency)


def analyze_files(
    files: list[Path],
    ssa_window: int = 256,
    ssa_components: int = SSA_COMPONENTS,
    signal_column: int = DEFAULT_SIGNAL_COLUMN,
) -> tuple[list[float], list[float], list[tuple[Path, np.ndarray, np.ndarray]]]:
    """Считывает сигналы и возвращает частоты FFT, SSA и данные для графика спектра."""

    fourier_frequencies: list[float] = []
    ssa_frequencies: list[float] = []
    traces: list[tuple[Path, np.ndarray, np.ndarray]] = []
    total_files = len(files)

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{total_files}] Обработка {path}")
        try:
            signal, file_sampling_frequency = read_signal(path, signal_column)
            window = min(ssa_window, signal.size)
            fourier_frequency, _, frequencies, magnitudes = dominant_frequency_with_magnitude(
                signal,
                file_sampling_frequency,
            )
            ssa_frequency = dominant_ssa_frequency(
                signal,
                file_sampling_frequency,
                window,
                ssa_components,
            )
        except ValueError as error:
            print(f"Пропуск {path}: {error}")
            continue

        fourier_frequencies.append(fourier_frequency)
        ssa_frequencies.append(ssa_frequency)
        traces.append((path, frequencies, magnitudes))

    return fourier_frequencies, ssa_frequencies, traces

