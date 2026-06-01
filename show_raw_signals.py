import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR, SAMPLING_FREQUENCY, signal_files

SPECTRUM_PLOTS = 4


def main() -> None:
    files = signal_files()
    if not files:
        raise FileNotFoundError("Не найдено файлов d####.")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= min(SPECTRUM_PLOTS, len(files)):
            axis.axis("off")
            continue
        path = files[index]
        signal = np.fromfile(path, dtype="<i2").astype(float)
        time = np.arange(signal.size) / SAMPLING_FREQUENCY
        axis.plot(time, signal, linewidth=0.8)
        axis.set_title(path.stem, fontsize=9)
        axis.set_xlabel("Время, с")
        axis.set_ylabel("Амплитуда")
        axis.grid(True, alpha=0.3)

    figure.suptitle("Исходные сигналы, первые 4 файла", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "raw_signals_preview.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
