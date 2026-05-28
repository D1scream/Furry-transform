from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pulse_analysis.analysis import FC_LOW, analyze_files

INPUT_DIR = Path("signals/Data . Code")
FILE_PATTERN = "**/*_ppg_ecg.csv"
SSA_WINDOW = 256
SSA_COMPONENTS = 4
SIGNAL_COLUMN = 1
TIME_SERIES_PLOTS = 4


def main() -> None:
    """Запускает анализ сигналов."""

    signal_files = sorted(path for path in INPUT_DIR.glob(FILE_PATTERN) if path.is_file())
    if not signal_files:
        raise FileNotFoundError(f"В директории {INPUT_DIR} не найдено файлов {FILE_PATTERN}.")

    fourier_frequencies, ssa_frequencies, traces = analyze_files(
        files=signal_files,
        ssa_window=SSA_WINDOW,
        ssa_components=SSA_COMPONENTS,
        signal_column=SIGNAL_COLUMN,
    )
    if not fourier_frequencies:
        raise RuntimeError("Не удалось найти ни одного корректного сигнала для анализа.")

    sample_traces = traces[:TIME_SERIES_PLOTS]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)

    for index, axis in enumerate(axes.flat):
        if index >= len(sample_traces):
            axis.axis("off")
            continue
        path, signal, sampling_frequency, _, _ = sample_traces[index]
        time = np.arange(signal.size) / sampling_frequency
        axis.plot(time, signal, linewidth=0.8)
        axis.set_title(path.stem, fontsize=9)
        axis.set_xlabel("Время, с")
        axis.set_ylabel("Амплитуда")
        axis.grid(True, alpha=0.3)

    figure.suptitle("PPG1 (преобразованный), первые 4 сигнала", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(sample_traces):
            axis.axis("off")
            continue
        path, _, _, frequencies, magnitudes = sample_traces[index]
        mask = frequencies <= FC_LOW
        axis.plot(frequencies[mask], magnitudes[mask], linewidth=0.8)
        axis.axvline(fourier_frequencies[index], color="r", linestyle="--", linewidth=1)
        axis.set_title(path.stem, fontsize=9)
        axis.set_xlabel("Частота, Гц")
        axis.set_ylabel("|FFT|")
        axis.grid(True, alpha=0.3)
    figure.suptitle("Спектр, первые 4 сигнала (красная — пик Фурье)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()

    lower_bound = min(min(fourier_frequencies), min(ssa_frequencies))
    upper_bound = max(max(fourier_frequencies), max(ssa_frequencies))

    plt.figure(figsize=(8, 8))
    plt.scatter(fourier_frequencies, ssa_frequencies, s=24, alpha=0.7, label="Сигналы")
    plt.plot([lower_bound, upper_bound], [lower_bound, upper_bound], "r--", label="y = x")
    plt.xlabel("Частота Фурье, Гц")
    plt.ylabel("Частота SSA, Гц")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Частоты Фурье:", fourier_frequencies)
    print("Частоты SSA:", ssa_frequencies)


if __name__ == "__main__":
    main()
