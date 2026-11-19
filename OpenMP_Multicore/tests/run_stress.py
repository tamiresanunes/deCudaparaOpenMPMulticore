"""Bateria isolada de testes OpenMP: malha, threads, tempo e limite prático.

Execute da raiz do projeto com:
    python OpenMP_Multicore/tests/run_stress.py

Cada caso usa start_mode=0, grava seus próprios arquivos e tem um tempo limite.
O primeiro problema em uma escada de malhas indica apenas o limite observado
nessa máquina, com esta entrada, número de threads e tempo limite.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "tests" / "smoke" / "input"
FILES = (
    ("data/output_variables.dat", 7),
    ("data/restart/restartU.dat", 1),
    ("data/restart/restartV.dat", 1),
    ("data/restart/restartPTC.dat", 3),
)
SUMMARY_FIELDS = (
    "case", "imax", "jmax", "threads", "itc_max", "final_time", "wall_seconds",
    "status", "exit_code", "output_rows", "max_abs_u", "max_abs_v", "max_abs_p",
    "max_abs_T", "max_abs_C", "min_T", "max_T", "min_C", "max_C", "folder",
)


@dataclass(frozen=True)
class Case:
    name: str
    imax: int
    threads: int
    final_time: float
    itc_max: int = 10


def prepare(case: Case, run_root: Path) -> Path:
    """Cria um diretório por execução sem alterar os dados do teste smoke."""
    folder = run_root / case.name
    (folder / "input").mkdir(parents=True)
    (folder / "data" / "restart").mkdir(parents=True)
    lines = (INPUT / "iterations.dat").read_text(encoding="utf-8").splitlines()
    if len(lines) < 11:
        raise ValueError("iterations.dat precisa conter 11 valores")
    lines[7] = f"{case.final_time:.12g}"
    lines[10] = "0"  # Condições iniciais: nunca carregar restart.
    (folder / "input" / "iterations.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(INPUT / "reference.dat", folder / "input" / "reference.dat")
    return folder


def inspect_output(folder: Path, imax: int, stdout: Path) -> dict[str, object]:
    """Exige arquivos completos e finitos; registra amplitudes para revelar crescimento."""
    log = stdout.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"MESH_CHECK imax=(\d+) jmax=(\d+)", log)
    if not match or int(match.group(1)) != imax:
        raise ValueError("MESH_CHECK ausente ou malha incorreta")
    if "FINAL_CHECK" not in log:
        raise ValueError("FINAL_CHECK ausente")
    jmax = int(match.group(2))
    expected = (imax * jmax, (imax + 1) * jmax, imax * (jmax + 1), imax * jmax)
    maxima = [0.0] * 5
    min_t = min_c = math.inf
    max_t = max_c = -math.inf
    for (relative, columns), rows_expected in zip(FILES, expected):
        path = folder / relative
        if not path.is_file():
            raise ValueError(f"arquivo ausente: {relative}")
        rows = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                values = [float(part) for part in line.split()]
                if len(values) != columns or not all(math.isfinite(v) for v in values):
                    raise ValueError(f"valor não finito ou colunas incorretas: {relative}, linha {rows + 1}")
                if relative == FILES[0][0]:
                    for i, value in enumerate(values[2:]):
                        maxima[i] = max(maxima[i], abs(value))
                    min_t, max_t = min(min_t, values[5]), max(max_t, values[5])
                    min_c, max_c = min(min_c, values[6]), max(max_c, values[6])
                rows += 1
        if rows != rows_expected:
            raise ValueError(f"{relative}: {rows} linhas, esperado {rows_expected}")
    return {"jmax": jmax, "output_rows": expected[0], **dict(zip(
        ("max_abs_u", "max_abs_v", "max_abs_p", "max_abs_T", "max_abs_C"), maxima)),
        "min_T": min_t, "max_T": max_t, "min_C": min_c, "max_C": max_c}


def run_case(case: Case, run_root: Path, exe: Path, timeout: int) -> dict[str, object]:
    """Executa um caso com prazo, preserva logs e classifica falhas separadamente."""
    folder = prepare(case, run_root)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(case.threads)
    if os.name == "nt":
        ucrt = Path("C:/msys64/ucrt64/bin")
        if ucrt.is_dir():
            env["PATH"] = str(ucrt) + os.pathsep + env.get("PATH", "")
    stdout = folder / "stdout.log"
    stderr = folder / "stderr.log"
    print(f"INICIO {case.name}: imax={case.imax}, threads={case.threads}, "
          f"tempo={case.final_time:g}, itc_max={case.itc_max}", flush=True)
    start = time.perf_counter()
    status = "ERROR"
    exit_code = ""
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            process = subprocess.run(
                [str(exe), str(case.imax), str(case.itc_max)], cwd=folder, env=env,
                stdout=out, stderr=err, timeout=timeout, check=False,
            )
            exit_code = process.returncode
            status = "PASS" if exit_code == 0 else "CRASH"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
        except OSError as error:
            err.write(str(error).encode("utf-8", errors="replace"))
            status = "LAUNCH_ERROR"
    result: dict[str, object] = {
        "case": case.name, "imax": case.imax, "jmax": "", "threads": case.threads,
        "itc_max": case.itc_max, "final_time": case.final_time,
        "wall_seconds": round(time.perf_counter() - start, 3),
        "status": status, "exit_code": exit_code, "output_rows": "",
        "max_abs_u": "", "max_abs_v": "", "max_abs_p": "",
        "max_abs_T": "", "max_abs_C": "", "min_T": "", "max_T": "",
        "min_C": "", "max_C": "", "folder": str(folder),
    }
    if status == "PASS":
        try:
            result.update(inspect_output(folder, case.imax, stdout))
            # Este fixture começa com T=1 e 0<=C<=1. Os limites amplos
            # sinalizam explosão numérica, não equivalência ou convergência.
            if (result["min_T"] < 0 or result["max_T"] > 2 or
                    result["min_C"] < -0.1 or result["max_C"] > 1.1):
                result["status"] = "UNSTABLE"
        except (OSError, ValueError) as error:
            result["status"] = "BAD_OUTPUT"
            with stderr.open("a", encoding="utf-8") as err:
                err.write(f"\nValidação: {error}\n")
    print(f"FIM {case.name}: {result['status']}, {result['wall_seconds']} s, "
          f"T=[{result['min_T']}, {result['max_T']}], "
          f"C=[{result['min_C']}, {result['max_C']}]", flush=True)
    return result


def same_outputs(first: Path, second: Path) -> bool:
    """Compara os quatro arquivos gravados, que têm precisão textual de seis casas."""
    for relative, _ in FILES:
        a = hashlib.sha256((first / relative).read_bytes()).digest()
        b = hashlib.sha256((second / relative).read_bytes()).digest()
        if a != b:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=120, help="segundos máximos por caso (padrão: 120)")
    parser.add_argument("--max-grid", type=int, default=512, help="maior imax da escada (padrão: 512)")
    parser.add_argument("--smoke", action="store_true", help="valida somente dois casos pequenos")
    args = parser.parse_args()
    if args.timeout < 1 or args.max_grid < 24:
        parser.error("timeout deve ser >= 1 e max-grid >= 24")
    exe = ROOT / "build" / ("cylinder_solver.exe" if os.name == "nt" else "cylinder_solver")
    if not exe.is_file():
        parser.error(f"executável ausente: {exe}; compile o projeto primeiro")
    run_root = ROOT / "build" / "stress_runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    run_root.mkdir(parents=True)
    summary = run_root / "summary.csv"
    cases = [Case("small_24_t1_1step", 24, 1, .005), Case("small_24_t2_1step", 24, 2, .005)]
    if not args.smoke:
        cases += [
            Case("medium_128_t1_1step", 128, 1, .005),
            Case("medium_128_t2_1step", 128, 2, .005),
            Case("medium_128_t4_1step", 128, 4, .005),
            Case("medium_128_t2_5steps", 128, 2, .025),
            Case("medium_128_t2_10steps", 128, 2, .05),
            Case("large_256_t2_1step", 256, 2, .005),
            Case("large_384_t2_itc2", 384, 2, .005, 2),
            Case("large_384_t2_itc5", 384, 2, .005, 5),
        ]
        cases += [Case(f"limit_{size}_t2_1step", size, 2, .005)
                  for size in (320, 352, 360, 364, 365, 366, 367, 384, 512, 768, 1024)
                  if size <= args.max_grid]
    results: dict[str, dict[str, object]] = {}
    last_limit_pass = "nenhum"
    first_limit_failure = "nenhum até o máximo configurado"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for case in cases:
            result = run_case(case, run_root, exe, args.timeout)
            results[case.name] = result
            writer.writerow(result)
            stream.flush()
            if case.name == "large_256_t2_1step" and result["status"] == "PASS":
                last_limit_pass = "256"
            if case.name.startswith("limit_"):
                if result["status"] == "PASS":
                    last_limit_pass = str(case.imax)
                else:
                    first_limit_failure = f"{case.imax} ({result['status']})"
                    break
    thread_mismatch = False
    try:
        a = results["small_24_t1_1step"]
        b = results["small_24_t2_1step"]
        if a["status"] == b["status"] == "PASS":
            equal = same_outputs(Path(str(a["folder"])), Path(str(b["folder"])))
            print("Saidas pequenas 1 vs 2 threads: " + ("IGUAIS" if equal else "DIFERENTES"))
            thread_mismatch |= not equal
        if not args.smoke:
            a = results["medium_128_t1_1step"]
            b = results["medium_128_t2_1step"]
            if a["status"] == b["status"] == "PASS":
                equal = same_outputs(Path(str(a["folder"])), Path(str(b["folder"])))
                print("Saidas medias 1 vs 2 threads: " + ("IGUAIS" if equal else "DIFERENTES"))
                thread_mismatch |= not equal
    except OSError as error:
        print(f"Comparação de threads indisponível: {error}")
    if not args.smoke:
        print(f"Escada de malha: ultimo PASS={last_limit_pass}; primeiro problema={first_limit_failure}")
        print("TIMEOUT indica limite de tempo configurado, nao necessariamente falha numerica.")
    print(f"Relatorio: {summary}")
    return 0 if not thread_mismatch and all(row["status"] == "PASS" for row in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
