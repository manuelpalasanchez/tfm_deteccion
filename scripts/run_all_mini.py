"""Entrenamiento + evaluacion mini (prueba e2e rapida) de los 4 modelos.

Uso:
    python scripts/run_all_mini.py
    python scripts/run_all_mini.py --eval-only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

CONFIGS = [
    "configs/mini_resnet50.yaml",
    "configs/mini_freqnet.yaml",
    "configs/mini_vit.yaml",
    "configs/mini_universalfakedetect.yaml",
]


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def _find_checkpoint(base_dir: Path, model_name: str) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = sorted(base_dir.glob(f"{model_name}_*"), reverse=True)
    for run_dir in candidates:
        ckpt = run_dir / "checkpoint_best.pth"
        if ckpt.exists():
            return ckpt
    return None


def _read_output_dir(config_path: Path) -> Path:
    import yaml
    sys.path.insert(0, str(ROOT))
    from utils.config import load_raw
    cfg = load_raw(config_path)
    return Path(cfg["output"]["base_dir"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    resultados = []

    for config in CONFIGS:
        config_path = ROOT / config
        model_name = config_path.stem.removeprefix("mini_")
        base_dir = _read_output_dir(config_path)
        print(f"\n{'='*60}\nModelo: {model_name}  ({config})\n{'='*60}")
        t0 = time.time()
        estado = "OK"

        if not args.eval_only:
            print(f"\n[TRAIN] {model_name}")
            rc = _run([sys.executable, "scripts/train.py", "--config", config])
            if rc != 0:
                estado = f"TRAIN ERROR ({rc})"
                resultados.append((model_name, estado, time.time() - t0))
                continue

        ckpt = _find_checkpoint(base_dir, model_name)
        if ckpt is None:
            estado = "SIN CHECKPOINT"
            resultados.append((model_name, estado, time.time() - t0))
            continue

        print(f"\n[EVAL] {model_name} <- {ckpt}")
        rc = _run(
            [sys.executable, "scripts/evaluate.py", "--config", config, "--checkpoint", str(ckpt)]
        )
        if rc != 0:
            estado = f"EVAL ERROR ({rc})"

        resultados.append((model_name, estado, time.time() - t0))

    print(f"\n{'='*60}\nResumen\n{'='*60}")
    for nombre, est, t in resultados:
        print(f"  {nombre:<22} {est:<22} {t:.1f}s")


if __name__ == "__main__":
    main()
