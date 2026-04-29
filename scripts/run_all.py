"""Lanza entrenamiento y evaluacion de los 4 modelos secuencialmente.

Uso:
    python scripts/run_all.py
    python scripts/run_all.py --eval-only   # Solo evaluacion (requiere checkpoints)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

# FreqNet en stand-by para la prueba final: se entrena desde cero y necesita
# mucho mas computo del que tenemos para llegar a un AUC competitivo. Volvera
# si hay margen tras consolidar los resultados de los otros tres.
CONFIGS = [
    "configs/resnet50.yaml",
    "configs/vit.yaml",
    "configs/universalfakedetect.yaml",
]


def _run(cmd: list[str]) -> int:
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def _find_checkpoint(model_name: str) -> Path | None:
    runs_dir = ROOT / "experiments" / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(runs_dir.glob(f"{model_name}_*"), reverse=True)
    for run_dir in candidates:
        ckpt = run_dir / "checkpoint_best.pth"
        if ckpt.exists():
            return ckpt
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Solo ejecuta evaluacion usando el ultimo checkpoint de cada modelo.",
    )
    args = parser.parse_args()

    resultados = []

    for config in CONFIGS:
        config_path = Path(config)
        model_name = config_path.stem
        print(f"\n{'='*60}")
        print(f"Modelo: {model_name}  ({config})")
        print(f"{'='*60}")
        t0 = time.time()
        estado = "OK"

        if not args.eval_only:
            print(f"\n[TRAIN] {model_name}")
            rc = _run([sys.executable, "scripts/train.py", "--config", config])
            if rc != 0:
                estado = f"TRAIN ERROR ({rc})"
                elapsed = time.time() - t0
                resultados.append((model_name, estado, elapsed))
                print(f"\n{model_name}: {estado}")
                continue

        ckpt = _find_checkpoint(model_name)
        if ckpt is None:
            estado = "SIN CHECKPOINT"
            elapsed = time.time() - t0
            resultados.append((model_name, estado, elapsed))
            print(f"\n{model_name}: no se encontro checkpoint en experiments/runs/")
            continue

        print(f"\n[EVAL] {model_name} <- {ckpt}")
        rc = _run(
            [sys.executable, "scripts/evaluate.py", "--config", config, "--checkpoint", str(ckpt)]
        )
        if rc != 0:
            estado = f"EVAL ERROR ({rc})"

        elapsed = time.time() - t0
        resultados.append((model_name, estado, elapsed))
        print(f"\n{model_name}: {estado} en {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("Resumen")
    print(f"{'='*60}")
    for nombre, est, t in resultados:
        print(f"  {nombre:<25} {est:<20} {t:.1f}s")


if __name__ == "__main__":
    main()
