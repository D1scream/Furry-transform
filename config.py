from pathlib import Path
import os

INPUT_DIR = Path("signals/АД Дудин")
OUTPUT_DIR = Path("outputs")

SAMPLING_FREQUENCY = 100.0
FC_HIGH = 0.5
FC_LOW = 5.0
SSA_WINDOW = 256
SSA_COMPONENTS = 4
SSA_CHUNKSIZE = 1

FILE_STRIDE = 10
SPECTRUM_PLOTS = 4
MAX_WORKERS = max(1, (os.cpu_count() or 2) // 2)

def signal_files(stride: int = 1) -> list[Path]:
    return sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.name.startswith("d") and path.name[1:].isdigit()
    )[::stride]
