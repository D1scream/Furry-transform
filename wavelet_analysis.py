from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt

from analysis import read_signal
from config import MAX_WORKERS, SAMPLING_FREQUENCY, WAVELET_OUTPUT_DIR, WAVELETS, signal_files


def _build_dwt_spectrum(signal: np.ndarray, wavelet_name: str) -> tuple[np.ndarray, list[str]]:
    wavelet = pywt.Wavelet(wavelet_name)
    level_count = max(1, min(8, pywt.dwt_max_level(signal.size, wavelet.dec_len)))
    coefficients = pywt.wavedec(signal, wavelet, level=level_count)

    detail_rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, detail in enumerate(coefficients[1:], start=1):
        source_x = np.linspace(0.0, 1.0, detail.size)
        target_x = np.linspace(0.0, 1.0, signal.size)
        detail_rows.append(np.interp(target_x, source_x, np.abs(detail)))
        labels.append(f"D{level_count - index + 1}")

    if detail_rows:
        return np.vstack(detail_rows), labels
    return np.zeros((1, signal.size), dtype=float), ["D1"]


def _save_wavelet_spectrum(
    signal_path: Path,
    signal: np.ndarray,
    wavelet_name: str,
    wavelet_dir: Path,
) -> None:
    spectrum, labels = _build_dwt_spectrum(signal, wavelet_name)
    time = np.arange(signal.size) / SAMPLING_FREQUENCY

    figure, axis = plt.subplots(figsize=(12, 5))
    image = axis.imshow(
        spectrum,
        aspect="auto",
        origin="lower",
        extent=(time[0], time[-1], 0, spectrum.shape[0]),
        cmap="viridis",
    )
    axis.set_title(f"{signal_path.stem} | wavelet={wavelet_name}")
    axis.set_xlabel("Время, с")
    axis.set_ylabel("Уровни DWT")
    axis.set_yticks(np.arange(0.5, spectrum.shape[0] + 0.5))
    axis.set_yticklabels(labels)
    figure.colorbar(image, ax=axis, label="|коэффициенты|")
    figure.tight_layout()
    figure.savefig(wavelet_dir / f"{signal_path.stem}.png", dpi=150)
    plt.close(figure)


def _process_signal(path: Path) -> tuple[Path, int, int]:
    try:
        signal = read_signal(path)
    except ValueError:
        return path, 0, len(WAVELETS)

    success = 0
    skipped = 0
    for wavelet_name in WAVELETS:
        try:
            _save_wavelet_spectrum(path, signal, wavelet_name, WAVELET_OUTPUT_DIR / wavelet_name)
            success += 1
        except ValueError:
            skipped += 1
    return path, success, skipped


def main() -> None:
    files = signal_files()
    if not files:
        raise FileNotFoundError("Не найдено файлов d####.")

    for wavelet_name in WAVELETS:
        (WAVELET_OUTPUT_DIR / wavelet_name).mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped_count = 0
    with ProcessPoolExecutor(max_workers=max(1, MAX_WORKERS)) as pool:
        for i, (path, success, skipped) in enumerate(pool.map(_process_signal, files), 1):
            success_count += success
            skipped_count += skipped
            print(f"[{i}/{len(files)}] {path.name}")

    print(
        f"Готово. Спектры в {WAVELET_OUTPUT_DIR}. "
        f"Успешно: {success_count}, пропущено: {skipped_count}"
    )


if __name__ == "__main__":
    main()
