import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import analyze_files
from config import FC_LOW, FILE_STRIDE, OUTPUT_DIR, SPECTRUM_PLOTS, signal_files




def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = signal_files(stride=FILE_STRIDE)
    if not files:
        raise FileNotFoundError("В директории не найдено файлов d####.")

    results = analyze_files(files)
    if not results:
        raise RuntimeError("Не удалось найти ни одного корректного сигнала для анализа.")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= min(SPECTRUM_PLOTS, len(results)):
            axis.axis("off")
            continue
        row = results[index]
        mask = row["freqs"] <= FC_LOW
        axis.plot(row["freqs"][mask] * 60, row["mags"][mask], linewidth=0.8)
        axis.axvline(row["fft_hz"] * 60, color="r", linestyle="--", linewidth=1)
        axis.set_title(row["path"].stem, fontsize=9)
        axis.set_xlabel("Частота, BPM")
        axis.set_ylabel("|FFT|")
        axis.grid(True, alpha=0.3)
    figure.suptitle("Спектр, первые 4 сигнала (красная — пик Фурье в BPM)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    spectrum_path = OUTPUT_DIR / "fft_spectra_preview.png"
    figure.savefig(spectrum_path, dpi=150)
    plt.close(figure)

    fft_bpm = [row["fft_hz"] * 60 for row in results]
    ssa_bpm = [row["ssa_hz"] * 60 for row in results]
    lims = [min(min(fft_bpm), min(ssa_bpm)), max(max(fft_bpm), max(ssa_bpm))]

    figure = plt.figure(figsize=(8, 8))
    plt.scatter(fft_bpm, ssa_bpm, s=24, alpha=0.8)
    plt.plot(lims, lims, "r--")
    plt.xlabel("Частота Фурье, BPM")
    plt.ylabel("Частота SSA, BPM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    comparison_path = OUTPUT_DIR / "fft_vs_ssa_bpm.png"
    figure.savefig(comparison_path, dpi=150)
    plt.close(figure)

    print(f"Графики сохранены: {spectrum_path}, {comparison_path}")


if __name__ == "__main__":
    main()
