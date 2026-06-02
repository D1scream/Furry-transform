# Сравнение FFT и SSA на датасете АД Дудин

## Запуск (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\fft_ssa_analysis.py
```

Если окружение уже создано:

```powershell
.\.venv\Scripts\python.exe .\fft_ssa_analysis.py
```

Просмотр исходных сигналов:

```powershell
.\.venv\Scripts\python.exe .\show_raw_signals.py
```

Пошаговый разбор в Jupyter:

```powershell
jupyter notebook signal_pipeline.ipynb
```

Вейвлет-спектры (DWT):

```powershell
.\.venv\Scripts\python.exe .\wavelet_analysis.py
```
