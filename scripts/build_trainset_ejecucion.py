"""Construye el 'trainset de ejecucion' a partir del zip de progan_train.

Criterio (decidido tras inspeccionar progan_train con scan_cnndetection.py):
las 20 categorias estan perfectamente balanceadas (~36k imgs cada una, 50/50
real/fake), por lo que el muestreo se reduce a un N fijo por categoria con
balance 50/50 dentro de cada una. Misma particion (semilla fija) para los 3
modelos activos (ResNet-50, ViT-B/16, UFD).

Uso tipico (Colab):
    python scripts/build_trainset_ejecucion.py \
        --zip /content/drive/MyDrive/cnndetection-datasets/progan_train.zip \
        --out /content/cnndetection/progan_train \
        --n-per-cat 5000 \
        --seed 42

Salida: <out>/<categoria>/{0_real,1_fake}/<archivo>.<ext>, lista para
CNNDetectionDataset con split='train'.
"""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REAL_DIRS = {"0_real", "nature"}
FAKE_DIRS = {"1_fake", "ai"}


def _index_zip(zip_path: Path) -> dict[tuple[str, str], list[zipfile.ZipInfo]]:
    """Devuelve {(categoria, label): [ZipInfo, ...]} a partir del indice del zip."""
    grouped: dict[tuple[str, str], list[zipfile.ZipInfo]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in VALID_EXTS:
                continue
            parts = [p for p in info.filename.replace("\\", "/").split("/") if p]
            if len(parts) < 3:
                continue
            label_idx = None
            for i, p in enumerate(parts):
                pl = p.lower()
                if pl in REAL_DIRS or pl in FAKE_DIRS:
                    label_idx = i
                    break
            if label_idx is None or label_idx == 0:
                continue
            categoria = parts[label_idx - 1]
            label = "0_real" if parts[label_idx].lower() in REAL_DIRS else "1_fake"
            grouped[(categoria, label)].append(info)
    return grouped


def _extract(
    zip_path: Path,
    out_dir: Path,
    chosen: dict[tuple[str, str], list[zipfile.ZipInfo]],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        for (cat, label), infos in chosen.items():
            target = out_dir / cat / label
            target.mkdir(parents=True, exist_ok=True)
            for info in infos:
                dest = target / Path(info.filename).name
                if dest.exists():
                    written += 1
                    continue
                with zf.open(info) as src, dest.open("wb") as dst:
                    dst.write(src.read())
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path, help="Ruta al zip de progan_train.")
    parser.add_argument("--out", required=True, type=Path, help="Directorio de salida.")
    parser.add_argument(
        "--n-per-cat",
        type=int,
        default=5000,
        help="Imagenes totales por categoria (mitad real, mitad fake).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Si se indica, escribe un JSON con las rutas exactas elegidas.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime el plan; no extrae nada.",
    )
    args = parser.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"No existe el zip: {args.zip}")
    if args.n_per_cat % 2 != 0:
        raise SystemExit("--n-per-cat debe ser par para balance 50/50 real/fake.")

    rng = random.Random(args.seed)
    per_label = args.n_per_cat // 2

    print(f"Indexando {args.zip} ...")
    grouped = _index_zip(args.zip)
    cats = sorted({c for (c, _l) in grouped.keys()})
    print(f"  categorias: {len(cats)} ({', '.join(cats)})")

    print(f"\nPlan (n_per_cat={args.n_per_cat}, seed={args.seed}):")
    print(f"{'categoria':<20} {'real':>8} {'fake':>8} {'subset':>10}")
    print("-" * 50)

    chosen: dict[tuple[str, str], list[zipfile.ZipInfo]] = {}
    grand_total = 0
    for cat in cats:
        reales = grouped.get((cat, "0_real"), [])
        fakes = grouped.get((cat, "1_fake"), [])
        n_r = min(per_label, len(reales))
        n_f = min(per_label, len(fakes))
        chosen[(cat, "0_real")] = rng.sample(reales, n_r) if n_r else []
        chosen[(cat, "1_fake")] = rng.sample(fakes, n_f) if n_f else []
        print(f"{cat:<20} {n_r:>8} {n_f:>8} {n_r + n_f:>10}")
        grand_total += n_r + n_f
    print("-" * 50)
    print(f"{'TOTAL':<20} {'':>8} {'':>8} {grand_total:>10}")

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "zip": str(args.zip),
            "out": str(args.out),
            "n_per_cat": args.n_per_cat,
            "seed": args.seed,
            "actual_total": grand_total,
            "per_category": {
                cat: {
                    "0_real": [i.filename for i in chosen.get((cat, "0_real"), [])],
                    "1_fake": [i.filename for i in chosen.get((cat, "1_fake"), [])],
                }
                for cat in cats
            },
        }
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"\nManifest escrito en {args.manifest}")

    if args.dry_run:
        print("\n[dry-run] no se extrae nada.")
        return

    print(f"\nExtrayendo a {args.out} ...")
    written = _extract(args.zip, args.out, chosen)
    print(f"OK - {written} imagenes escritas.")


if __name__ == "__main__":
    main()
