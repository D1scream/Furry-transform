from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = Path("signals/Data . Code")
FILE_PATTERN = "**/*_ppg_ecg.csv"
SIGNAL_COLUMN = 2
PLOTS_COUNT = 4


def read_raw_signal(path: Path, signal_column: int) -> tuple[np.ndarray, np.ndarray]:
    """Читает временную ось и исходный сигнал из CSV без препроцессинга."""

    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] <= signal_column:
        raise ValueError(f"Файл {path} не содержит колонку сигнала с индексом {signal_column}.")
    time = np.asarray(data[:, 0], dtype=float)
    signal = np.asarray(data[:, signal_column], dtype=float)
    return time, signal


def main() -> None:
    """Показывает графики исходных сигналов."""

    signal_files = sorted(path for path in INPUT_DIR.glob(FILE_PATTERN) if path.is_file())
    if not signal_files:
        raise FileNotFoundError(f"В директории {INPUT_DIR} не найдено файлов {FILE_PATTERN}.")

    sample_files = signal_files[:PLOTS_COUNT]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)

    for index, axis in enumerate(axes.flat):
        if index >= len(sample_files):
            axis.axis("off")
            continue
        path = sample_files[index]
        try:
            time, signal = read_raw_signal(path, SIGNAL_COLUMN)
        except ValueError as error:
            axis.text(0.5, 0.5, str(error), ha="center", va="center", wrap=True)
            axis.axis("off")
            continue

        axis.plot(time, signal, linewidth=0.8)
        axis.set_title(path.stem, fontsize=9)
        axis.set_xlabel("Время, с")
        axis.set_ylabel("Амплитуда")
        axis.grid(True, alpha=0.3)

    figure.suptitle("Исходные сигналы ECG, первые 4 файла", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show(block=True)


if __name__ == "__main__":
    main()
