from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pulse_analysis.analysis import analyze_files

INPUT_DIR = Path("signals/Data . Code")
FILE_PATTERN = "**/*_ppg_ecg.csv"
SIGNAL_LENGTH = None
SSA_WINDOW = 256
SSA_COMPONENTS = 4
MIN_FREQUENCY = 0.5
SIGNAL_COLUMN = 1


def main() -> None:
    """Запускает анализ сигналов."""

    signal_files = sorted(path for path in INPUT_DIR.glob(FILE_PATTERN) if path.is_file())
    if not signal_files:
        raise FileNotFoundError(f"В директории {INPUT_DIR} не найдено файлов {FILE_PATTERN}.")

    fourier_frequencies, ssa_frequencies, traces = analyze_files(
        files=signal_files,
        signal_length=SIGNAL_LENGTH,
        ssa_window=SSA_WINDOW,
        ssa_components=SSA_COMPONENTS,
        min_frequency=MIN_FREQUENCY,
        signal_column=SIGNAL_COLUMN,
    )
    if not fourier_frequencies:
        raise RuntimeError("Не удалось найти ни одного корректного сигнала для анализа.")

    plots_count = len(traces)
    columns = 4
    rows = (plots_count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, 3 * rows), squeeze=False)

    for index, (path, signal, sampling_frequency) in enumerate(traces):
        time = np.arange(signal.size) / sampling_frequency
        axis = axes[index // columns][index % columns]
        axis.plot(time, signal, linewidth=0.8)
        axis.set_title(path.stem, fontsize=9)
        axis.set_xlabel("Время, с")
        axis.set_ylabel("Амплитуда")
        axis.grid(True, alpha=0.3)

    for index in range(plots_count, rows * columns):
        axes[index // columns][index % columns].axis("off")

    figure.suptitle("PPG1 (преобразованный)", fontsize=14)
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
