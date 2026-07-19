@echo off
REM run_experiments.bat
REM
REM Batch runner for Hybrid Parallel BFS experiments on Windows (MS-MPI).
REM
REM Prerequisites:
REM   - MS-MPI installed (https://github.com/microsoft/Microsoft-MPI)
REM   - bin\bfs.exe and bin\graph_gen.exe built
REM   - Graph files generated in data\ (run `make graphs` or generate manually)
REM
REM Output: results\timing.csv

setlocal enabledelayedexpansion

set BFS_EXE=bin\bfs.exe
set GRAPHS_DIR=data
set RESULTS_DIR=results
set CSV_FILE=%RESULTS_DIR%\timing.csv
set NRUNS=5

if not exist %BFS_EXE% (
    echo ERROR: %BFS_EXE% not found. Build first with: make all
    exit /b 1
)

if not exist %RESULTS_DIR% mkdir %RESULTS_DIR%

REM Write CSV header
echo graph_size,mode,np,threads,total_cores,run_id,time_ms,levels,visited > %CSV_FILE%

REM Graph sizes to test
set SIZES=20k 40k 80k 160k
set VERTICES_20k=20000
set VERTICES_40k=40000
set VERTICES_80k=80000
set VERTICES_160k=160000

REM Pure MPI configurations (np)
set MPI_NP=1 2 4 8

REM Hybrid configurations (np x threads)
REM We'll run: 1x2, 1x4, 2x2, 2x4, 4x2
set HYBRID_CFG=1,2 1,4 2,2 2,4 4,2

echo ============================================
echo  Hybrid Parallel BFS — Experiment Runner
echo ============================================
echo Runs per config: %NRUNS%
echo Output: %CSV_FILE%
echo.

for %%S in (%SIZES%) do (
    set GRAPH=%GRAPHS_DIR%\graph_%%S.csr
    if not exist !GRAPH! (
        echo WARNING: !GRAPH! not found, skipping...
        echo.
        goto :next_size
    )

    REM Get vertex count
    if "%%S"=="20k"  set V=%VERTICES_20k%
    if "%%S"=="40k"  set V=%VERTICES_40k%
    if "%%S"=="80k"  set V=%VERTICES_80k%
    if "%%S"=="160k" set V=%VERTICES_160k%

    echo ---- Graph: %%S vertices (%V%) ----

    REM ==========================================
    REM Pure MPI runs
    REM ==========================================
    for %%N in (%MPI_NP%) do (
        echo   Pure MPI: np=%%N
        for /L %%R in (1,1,%NRUNS%) do (
            mpiexec -np %%N %BFS_EXE% --mode mpi --graph !GRAPH! --root 0 --csv 2>nul | findstr "RESULT:" >> tmp_result.txt
        )
    )

    REM ==========================================
    REM Hybrid MPI+OpenMP runs
    REM ==========================================
    for %%C in (%HYBRID_CFG%) do (
        for /F "tokens=1,2 delims=," %%A in ("%%C") do (
            set NP=%%A
            set TH=%%B
            echo   Hybrid: np=!NP! x threads=!TH!
            for /L %%R in (1,1,%NRUNS%) do (
                mpiexec -np !NP! %BFS_EXE% --mode hybrid --graph !GRAPH! --root 0 --threads !TH! --csv 2>nul | findstr "RESULT:" >> tmp_result.txt
            )
        )
    )

    :next_size
    echo.
)

REM Parse tmp_result.txt and build the final CSV
if exist tmp_result.txt (
    for /F "tokens=2 delims=:" %%L in (tmp_result.txt) do (
        REM Parse: mode,np,threads,V,E,time_ms,levels,visited
        for /F "tokens=1-8 delims=," %%a in ("%%L") do (
            set MODE=%%a
            set NP=%%b
            set TH=%%c
            set VV=%%d
            set EE=%%e
            set TIME=%%f
            set LVL=%%g
            set VIS=%%h

            REM Determine graph_size label
            if !VV!==20000  set GS=20k
            if !VV!==40000  set GS=40k
            if !VV!==80000  set GS=80k
            if !VV!==160000 set GS=160k

            REM Compute total_cores = np * threads
            set /A TC=!NP! * !TH!

            REM We don't have run_id from the tool but we can number them
            REM For simplicity, output what we have
            echo !GS!,!MODE!,!NP!,!TH!,!TC!,0,!TIME!,!LVL!,!VIS! >> %CSV_FILE%
        )
    )
    del tmp_result.txt
)

echo Done! Results saved to %CSV_FILE%
endlocal
