"""Compara campos CUDA e OpenMP produzidos com a mesma entrada e malha.

Uso: python compare_outputs.py PASTA_CUDA PASTA_OPENMP [--atol 2e-6] [--rtol 1e-5]
As pastas devem conter data/output_variables.dat e data/restart/*.dat.
"""

import argparse
import math
from pathlib import Path
import sys


FILES = {
    "data/output_variables.dat": ("x", "y", "u", "v", "p", "T", "C"),
    "data/restart/restartU.dat": ("U_face",),
    "data/restart/restartV.dat": ("V_face",),
    "data/restart/restartPTC.dat": ("p", "T", "C"),
}


def read_table(path: Path, columns: tuple[str, ...]) -> list[list[float]]:
    """Lê linhas numéricas e rejeita arquivos truncados ou valores não finitos."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = [float(value) for value in line.split()]
            if len(row) != len(columns):
                raise ValueError(f"{path}:{number}: esperado {len(columns)} colunas, recebido {len(row)}")
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"{path}:{number}: NaN ou infinito")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: arquivo vazio")
    return rows


def compare(cuda_dir: Path, omp_dir: Path, atol: float, rtol: float) -> bool:
    """Reporta erros por campo e aprova apenas se todas as células respeitam a tolerância."""
    passed = True
    for relative, fields in FILES.items():
        cuda_rows = read_table(cuda_dir / relative, fields)
        omp_rows = read_table(omp_dir / relative, fields)
        if len(cuda_rows) != len(omp_rows):
            print(f"FALHOU {relative}: linhas CUDA={len(cuda_rows)}, OpenMP={len(omp_rows)}")
            passed = False
            continue
        print(f"\n{relative}: {len(cuda_rows)} linhas")
        for column, field in enumerate(fields):
            differences = [abs(a[column] - b[column]) for a, b in zip(cuda_rows, omp_rows)]
            failures = [index for index, (a, b) in enumerate(zip(cuda_rows, omp_rows))
                        if abs(a[column] - b[column]) > atol + rtol * max(abs(a[column]), abs(b[column]))]
            worst = max(range(len(differences)), key=differences.__getitem__)
            max_abs = differences[worst]
            rms = math.sqrt(sum(delta * delta for delta in differences) / len(differences))
            status = "OK" if not failures else "FALHOU"
            print(f"  {field:7s} {status:6s} max_abs={max_abs:.6g} RMS={rms:.6g} "
                  f"fora={len(failures)}/{len(differences)} "
                  f"pior_linha={worst + 1} CUDA={cuda_rows[worst][column]:.9g} "
                  f"OpenMP={omp_rows[worst][column]:.9g}")
            passed &= not failures
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cuda", type=Path, help="Pasta de execução CUDA")
    parser.add_argument("openmp", type=Path, help="Pasta de execução OpenMP")
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("as tolerâncias devem ser não negativas")
    try:
        passed = compare(args.cuda, args.openmp, args.atol, args.rtol)
    except (OSError, ValueError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2
    print("\nRESULTADO: " + ("EQUIVALENTES dentro das tolerancias" if passed else "DIFERENTES"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
