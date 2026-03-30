@echo off
setlocal

set VENV=.venv\Scripts
set OUT=bench_results\rtx5090_en
set WLK=%VENV%\wlk.exe

if not exist %OUT% mkdir %OUT%

echo Running all benchmarks to %OUT%...
echo.

echo [1/13] fw LA base
%WLK% bench --backend faster-whisper --model base --languages en --json %OUT%\faster-whisper_base_localagreement.json --backend-policy localagreement
echo.

echo [2/13] fw SS base
%WLK% bench --backend faster-whisper --model base --languages en --json %OUT%\faster-whisper_base_simulstreaming.json --backend-policy simulstreaming
echo.

echo [3/13] fw LA small
%WLK% bench --backend faster-whisper --model small --languages en --json %OUT%\faster-whisper_small_localagreement.json --backend-policy localagreement
echo.

echo [4/13] fw SS small
%WLK% bench --backend faster-whisper --model small --languages en --json %OUT%\faster-whisper_small_simulstreaming.json --backend-policy simulstreaming
echo.

echo [5/13] fw LA large-v3
%WLK% bench --backend faster-whisper --model large-v3 --languages en --json %OUT%\faster-whisper_large-v3_localagreement.json --backend-policy localagreement
echo.

echo [6/13] fw SS large-v3
%WLK% bench --backend faster-whisper --model large-v3 --languages en --json %OUT%\faster-whisper_large-v3_simulstreaming.json --backend-policy simulstreaming
echo.

echo [7/13] fw LA turbo
%WLK% bench --backend faster-whisper --model large-v3-turbo --languages en --json %OUT%\faster-whisper_large-v3-turbo_localagreement.json --backend-policy localagreement
echo.

echo [8/13] fw SS turbo
%WLK% bench --backend faster-whisper --model large-v3-turbo --languages en --json %OUT%\faster-whisper_large-v3-turbo_simulstreaming.json --backend-policy simulstreaming
echo.

echo [9/13] voxtral hf
%WLK% bench --backend voxtral --languages en --json %OUT%\voxtral_base.json
echo.

echo [10/13] qwen3 0.6B
%WLK% bench --backend qwen3 --model qwen3:0.6b --languages en --json %OUT%\qwen3_qwen3-0.6b.json
echo.

echo [11/13] qwen3 1.7B
%WLK% bench --backend qwen3 --model qwen3:1.7b --languages en --json %OUT%\qwen3_qwen3-1.7b.json
echo.

echo [12/13] qwen3-kv 0.6B
%WLK% bench --backend qwen3-simul-kv --model qwen3:0.6b --languages en --json %OUT%\qwen3-simul-kv_qwen3-0.6b.json
echo.

echo [13/13] qwen3-kv 1.7B
%WLK% bench --backend qwen3-simul-kv --model qwen3:1.7b --languages en --json %OUT%\qwen3-simul-kv_qwen3-1.7b.json
echo.

echo.
echo All benchmarks complete. Compiling results...
%VENV%\python scripts\run_all_benches.py --output-dir %OUT% --plot-only

echo Done!
