from pathlib import Path

import numpy as np


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


def read_signal(
    path: Path,
    signal_length: int | None = None,
    signal_column: int = 1,
) -> tuple[np.ndarray, float]:
    """Читает PPG-сигнал из CSV и оценивает частоту дискретизации по времени."""

    try:
        data = np.loadtxt(path, delimiter=",")
    except Exception as error:
        raise ValueError(f"Не удалось прочитать CSV-файл {path}: {error}") from error

    if data.ndim != 2 or data.shape[1] <= signal_column:
        raise ValueError(f"Файл {path} не содержит колонку сигнала с индексом {signal_column}.")

    time = np.asarray(data[:, 0], dtype=float)
    signal = np.asarray(data[:, signal_column], dtype=float)
    if signal_length is not None:
        time = time[:signal_length]
        signal = signal[:signal_length]

    if time.size < 2:
        raise ValueError(f"Файл {path} содержит слишком мало временных отсчетов.")
    time_steps = np.diff(time)
    median_step = float(np.median(time_steps))
    if median_step <= 0:
        raise ValueError(f"Некорректная временная шкала в файле {path}.")

    sampling_frequency = 1.0 / median_step
    _validate_signal(signal, sampling_frequency)
    return signal, sampling_frequency


def dominant_frequency_with_magnitude(
    signal: np.ndarray,
    sampling_frequency: float,
    min_frequency: float = 0.1,
) -> tuple[float, float]:
    """Возвращает частоту и амплитуду максимального пика в спектре сигнала."""

    _validate_signal(signal, sampling_frequency)
    centered_signal = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered_signal)
    frequencies = np.fft.rfftfreq(centered_signal.size, d=1.0 / sampling_frequency)
    magnitudes = np.abs(spectrum)
    start_index = int(np.searchsorted(frequencies, min_frequency))
    if start_index >= magnitudes.size:
        raise ValueError(
            f"Минимальная частота {min_frequency} Гц выше диапазона спектра сигнала."
        )
    peak_index = start_index + int(np.argmax(magnitudes[start_index:]))
    return float(frequencies[peak_index]), float(magnitudes[peak_index])


def build_trajectory_matrix(signal: np.ndarray, window: int) -> np.ndarray:
    """Формирует траекторную матрицу для SSA."""

    return np.lib.stride_tricks.sliding_window_view(signal, window).T


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
    components_count: int = 4,
    min_frequency: float = 0.1,
) -> float:
    """Ищет основную частоту среди первых компонент SSA."""

    _validate_signal(signal, sampling_frequency)
    if not 2 <= window <= signal.size:
        raise ValueError(
            f"Окно SSA должно быть в диапазоне [2, {signal.size}], получено {window}."
        )

    centered_signal = signal - np.mean(signal)
    trajectory = build_trajectory_matrix(centered_signal, window)
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
        frequency, peak_magnitude = dominant_frequency_with_magnitude(
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
    signal_length: int | None = None,
    ssa_window: int = 256,
    ssa_components: int = 4,
    min_frequency: float = 0.1,
    signal_column: int = 1,
) -> tuple[list[float], list[float]]:
    """Считывает список сигналов и возвращает частоты FFT и SSA."""

    fourier_frequencies: list[float] = []
    ssa_frequencies: list[float] = []
    total_files = len(files)

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{total_files}] Обработка {path}")
        try:
            signal, file_sampling_frequency = read_signal(path, signal_length, signal_column)
            window = min(ssa_window, signal.size)
            fourier_frequency, _ = dominant_frequency_with_magnitude(
                signal,
                file_sampling_frequency,
                min_frequency,
            )
            ssa_frequency = dominant_ssa_frequency(
                signal,
                file_sampling_frequency,
                window,
                ssa_components,
                min_frequency,
            )
        except ValueError as error:
            print(f"Пропуск {path}: {error}")
            continue

        fourier_frequencies.append(fourier_frequency)
        ssa_frequencies.append(ssa_frequency)

    return fourier_frequencies, ssa_frequencies
