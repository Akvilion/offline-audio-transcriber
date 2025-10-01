# 🛠️ PowerShell-скрипт для підготовки середовища OpenAI Whisper на Windows (з uv)
# Версія: 1.2 (uv edition)
# Автор оригіналу: Михаил Шардин | Адаптація під Windows + uv: ChatGPT (для A12)
#
# Що робить:
# - Перевіряє/ставить Python, uv, ffmpeg
# - Створює venv через uv
# - Стає PyTorch (stable або nightly) через uv pip з правильним CUDA індексом
# - Стає openai-whisper та аудіо-залежності через uv pip
# - Гонить короткий GPU self-test (CUDA, torch, whisper)
#
# Як запускати:
#   1) Збережи файл як setup_whisper.ps1
#   2) One-time: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
#   3) Запусти: .\setup_whisper.ps1

$ErrorActionPreference = "Stop"

Write-Host "🚀 Установка середовища для OpenAI Whisper (Windows + uv)"
Write-Host "========================================================="

# --- ОС/довідкова інфа ---
Write-Host "📋 System info:"
try {
  (Get-ComputerInfo | Select-Object OsName,OsVersion,WindowsProductName) | Format-List
} catch { Write-Host " (skip) " }

# --- Python ---
Write-Host "`n🐍 Перевірка Python..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host "⚠️  Python не знайдено. Ставлю через winget..."
  winget install -e --id Python.Python.3.12 | Out-Null
}
python --version

# --- uv (надшвидкий менеджер) ---
Write-Host "`n⚡ Перевірка uv..."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "⚠️  uv не знайдено. Ставлю через winget..."
  winget install -e --id astral-sh.uv | Out-Null
  # Якщо одразу не підхопилося у PATH — підкажемо перезапустити PowerShell:
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "ℹ️  uv встановлено, але не у PATH для поточної сесії. Перезапусти PowerShell та запусти скрипт знову."
    exit 1
  }
}
uv --version

# --- ffmpeg ---
Write-Host "`n🎵 Перевірка ffmpeg..."
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Write-Host "⚠️  ffmpeg не знайдено. Ставлю через winget..."
  winget install -e --id Gyan.FFmpeg | Out-Null
}
ffmpeg -version | Select-String "ffmpeg version" | ForEach-Object { $_.Line }

# --- NVIDIA / CUDA ---
Write-Host "`n🎮 Перевірка NVIDIA драйверів..."
$HasNvidia = $false
$GPU_INFO = ""
if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
  $HasNvidia = $true
  $GPU_INFO = (& nvidia-smi --query-gpu=name --format=csv,noheader,nounits) -join "; "
  Write-Host ("✅ Знайдено NVIDIA GPU: {0}" -f $GPU_INFO)
} else {
  Write-Host "⚠️  NVIDIA драйвери не знайдено. Для GPU-режиму встанови Game Ready / Studio Driver з офсайту NVIDIA."
}

Write-Host "`n🔧 Перевірка CUDA Toolkit..."
$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $CudaRoot) {
  Write-Host "✅ CUDA Toolkit знайдено: $CudaRoot"
} else {
  Write-Host "ℹ️  CUDA Toolkit не обов'язково ставити вручну (PyTorch колеса містять CUDA runtime)."
  Write-Host "    Якщо треба повний SDK: https://developer.nvidia.com/cuda-downloads"
}

# --- venv через uv ---
Write-Host "`n🏠 Створення/активація virtualenv через uv..."
# створює .venv в поточній директорії (ідемпотентно)
uv venv
# активація для поточної сесії PowerShell
$venvActivate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
  . $venvActivate
  Write-Host "✅ Активовано .venv"
} else {
  Write-Host "❌ Не знайдено $venvActivate"
  exit 1
}

# --- Вибір індексу для PyTorch під GPU покоління ---
Write-Host "`n🔥 Встановлення PyTorch через uv pip..."
# За замовчуванням — стабільні колеса з CUDA 12.1 (мінімально конфліктні).
$TorchArgs = @("torch","torchvision","torchaudio","--index-url","https://download.pytorch.org/whl/cu121")

if ($HasNvidia) {
  # Якщо свіже покоління (RTX 50 / деякі 40 / 5060 Ti) — беремо nightly cu129
  if ($GPU_INFO -match "RTX 50|RTX 4090|RTX 4080|RTX 4070|RTX 4060|RTX 5060|RTX 40|RTX 5060 Ti") {
    Write-Host "🚀 Виявлено нове покоління GPU ⇒ ставимо PyTorch nightly (CUDA 12.9) через офіційний індекс."
    $TorchArgs = @("--pre","torch","torchvision","torchaudio","--index-url","https://download.pytorch.org/whl/nightly/cu129")
  } else {
    Write-Host "📦 Ставимо стабільні колеса PyTorch (CUDA 12.1)."
  }
} else {
  Write-Host "💻 GPU не виявлено ⇒ можна поставити CPU-білди (але залишимо дефолт; за потреби зміни індекс на https://download.pytorch.org/whl/cpu)."
}

# Власне інсталяція PyTorch
uv pip install @TorchArgs

# --- Whisper та супутні бібліотеки (через uv pip) ---
Write-Host "`n🎙️  Встановлення OpenAI Whisper..."
uv pip install openai-whisper

Write-Host "`n📚 Додаткові бібліотеки для аудіо..."
uv pip install numpy scipy librosa soundfile pydub

# --- Тест ---
Write-Host "`n🧪 Перевірка встановлення (torch + CUDA + whisper)..."
$testPy = @'
import torch, whisper, sys
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA (torch): {torch.version.cuda}")
    x = torch.zeros(4,4, device="cuda") + 1
    print("✅ CUDA tensor ok:", x.sum().item())
else:
    print("💻 CPU mode")
print("✅ whisper import ok:", whisper is not None)
'@
python - <<PYCODE
$testPy
PYCODE

Write-Host "`n🎉 Готово!"
Write-Host "========================================="
Write-Host "Активувати середовище у новій сесії: .\.venv\Scripts\Activate.ps1"
Write-Host "Приклад запуску:  python .\whisper_transcribe.py .\audio large .\results"
