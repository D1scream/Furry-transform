from pathlib import Path

import matplotlib.pyplot as plt

from pulse_analysis.analysis import FC_LOW, analyze_files

INPUT_DIR = Path("signals/Data . Code")
FILE_PATTERN = "**/*_ppg_ecg.csv"
SSA_WINDOW = 256
SSA_COMPONENTS = 4
SIGNAL_COLUMN = 2
SPECTRUM_PLOTS = 4
HZ_TO_BPM = 60.0


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
    fourier_bpm = [frequency * HZ_TO_BPM for frequency in fourier_frequencies]
    ssa_bpm = [frequency * HZ_TO_BPM for frequency in ssa_frequencies]

    sample_traces = traces[:SPECTRUM_PLOTS]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(sample_traces):
            axis.axis("off")
            continue
        path, frequencies, magnitudes = sample_traces[index]
        mask = frequencies <= FC_LOW
        axis.plot(frequencies[mask] * HZ_TO_BPM, magnitudes[mask], linewidth=0.8)
        axis.axvline(fourier_bpm[index], color="r", linestyle="--", linewidth=1)
        axis.set_title(path.stem, fontsize=9)
        axis.set_xlabel("Частота, BPM")
        axis.set_ylabel("|FFT|")
        axis.grid(True, alpha=0.3)
    figure.suptitle("Спектр, первые 4 сигнала (красная — пик Фурье в BPM)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()

    lower_bound = min(min(fourier_bpm), min(ssa_bpm))
    upper_bound = max(max(fourier_bpm), max(ssa_bpm))
    participant_ids = [path.parent.name for path, _, _ in traces]
    unique_participants = sorted(set(participant_ids))
    color_map = plt.get_cmap("tab10")

    plt.figure(figsize=(8, 8))
    for index, participant_id in enumerate(unique_participants):
        participant_indices = [
            point_index
            for point_index, current_id in enumerate(participant_ids)
            if current_id == participant_id
        ]
        participant_fourier = [fourier_bpm[point_index] for point_index in participant_indices]
        participant_ssa = [ssa_bpm[point_index] for point_index in participant_indices]
        plt.scatter(
            participant_fourier,
            participant_ssa,
            s=24,
            alpha=0.8,
            color=color_map(index % 10),
            label=participant_id,
        )
    plt.plot([lower_bound, upper_bound], [lower_bound, upper_bound], "r--", label="y = x")
    plt.xlabel("Частота Фурье, BPM")
    plt.ylabel("Частота SSA, BPM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
