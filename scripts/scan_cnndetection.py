"""Recuento del dataset CNNDetection (en disco o como zip).

Modo carpeta:
    python scripts/scan_cnndetection.py
    python scripts/scan_cnndetection.py --root data/datasets/cnndetection

Modo zip (no necesita descomprimir; lee el indice del zip):
    python scripts/scan_cnndetection.py --zip /content/drive/MyDrive/cnndetection-datasets/progan_train.zip

Salida CSV opcional:
    python scripts/scan_cnndetection.py --zip ... --csv reports/scan_progan_train.csv

Pensado como utilidad temporal previa a decidir el muestreo proporcional para
construir el "trainset de ejecucion" (~10-12k imgs sobre las 20 categorias).
"""

import argparse
import csv
import zipfile
from collections import defaultdict
from pathlib import Path

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REAL_DIRS = {"0_real", "nature"}
FAKE_DIRS = {"1_fake", "ai"}


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} {units[-1]}"


def _scan_category_dir(cat_dir: Path) -> tuple[int, int, int]:
    n_real = 0
    n_fake = 0
    total_bytes = 0
    for sub in cat_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if name in REAL_DIRS:
            label = "real"
        elif name in FAKE_DIRS:
            label = "fake"
        else:
            continue
        for f in sub.rglob("*"):
            if f.is_file() and f.suffix.lower() in VALID_EXTS:
                if label == "real":
                    n_real += 1
                else:
                    n_fake += 1
                try:
                    total_bytes += f.stat().st_size
                except OSError:
                    pass
    return n_real, n_fake, total_bytes


def _scan_split_dir(split_dir: Path) -> list[dict]:
    rows = []
    for cat_dir in sorted(split_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        n_real, n_fake, size = _scan_category_dir(cat_dir)
        if n_real == 0 and n_fake == 0:
            continue
        rows.append(
            {
                "split": split_dir.name,
                "categoria": cat_dir.name,
                "real": n_real,
                "fake": n_fake,
                "total": n_real + n_fake,
                "bytes": size,
                "tamano": _human_size(size),
            }
        )
    return rows


def _scan_zip(zip_path: Path) -> list[dict]:
    """Recorre el indice del zip sin extraer. Asume estructura
    [<prefix>/]<categoria>/{0_real|1_fake|nature|ai}/<archivo>.<ext>.
    """
    split_name = zip_path.stem
    agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"real": 0, "fake": 0, "bytes": 0}
    )

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in VALID_EXTS:
                continue

            parts = info.filename.replace("\\", "/").split("/")
            parts = [p for p in parts if p]
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
            label = "real" if parts[label_idx].lower() in REAL_DIRS else "fake"

            agg[categoria][label] += 1
            agg[categoria]["bytes"] += info.file_size

    rows = []
    for cat in sorted(agg.keys()):
        d = agg[cat]
        rows.append(
            {
                "split": split_name,
                "categoria": cat,
                "real": d["real"],
                "fake": d["fake"],
                "total": d["real"] + d["fake"],
                "bytes": d["bytes"],
                "tamano": _human_size(d["bytes"]),
            }
        )
    return rows


def _print_table(rows: list[dict], split: str) -> None:
    sub = [r for r in rows if r["split"] == split]
    if not sub:
        print(f"\n[{split}] sin categorias")
        return
    header = f"{'categoria':<20} {'real':>8} {'fake':>8} {'total':>10} {'tamano':>12}"
    print(f"\n=== {split} ===")
    print(header)
    print("-" * len(header))
    for r in sub:
        print(
            f"{r['categoria']:<20} {r['real']:>8} {r['fake']:>8} "
            f"{r['total']:>10} {r['tamano']:>12}"
        )
    total_real = sum(r["real"] for r in sub)
    total_fake = sum(r["fake"] for r in sub)
    total_imgs = sum(r["total"] for r in sub)
    total_bytes = sum(r["bytes"] for r in sub)
    print("-" * len(header))
    print(
        f"{'TOTAL':<20} {total_real:>8} {total_fake:>8} "
        f"{total_imgs:>10} {_human_size(total_bytes):>12}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Raiz del dataset CNNDetection en disco (contiene progan_train, progan_val, ...).",
    )
    src.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Ruta a un zip de CNNDetection (p. ej. progan_train.zip). No lo descomprime.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Si se indica, escribe el detalle por categoria en este CSV.",
    )
    args = parser.parse_args()

    if args.zip is None and args.root is None:
        args.root = Path("data/datasets/cnndetection")

    all_rows: list[dict] = []
    splits: list[str] = []

    if args.zip is not None:
        if not args.zip.exists():
            raise SystemExit(f"No existe el zip: {args.zip}")
        all_rows = _scan_zip(args.zip)
        splits = [args.zip.stem]
    else:
        if not args.root.exists():
            raise SystemExit(f"No existe la raiz: {args.root}")
        split_dirs = sorted(d for d in args.root.iterdir() if d.is_dir())
        if not split_dirs:
            raise SystemExit(f"Sin splits en {args.root}")
        for split_dir in split_dirs:
            all_rows.extend(_scan_split_dir(split_dir))
        splits = [d.name for d in split_dirs]

    for split in splits:
        _print_table(all_rows, split)

    print("\n=== Resumen global ===")
    total_real = sum(r["real"] for r in all_rows)
    total_fake = sum(r["fake"] for r in all_rows)
    total_imgs = sum(r["total"] for r in all_rows)
    total_bytes = sum(r["bytes"] for r in all_rows)
    print(f"  splits     : {len(splits)} ({', '.join(splits)})")
    print(f"  categorias : {len({(r['split'], r['categoria']) for r in all_rows})}")
    print(f"  imagenes   : {total_imgs} (real={total_real}, fake={total_fake})")
    print(f"  tamano     : {_human_size(total_bytes)}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["split", "categoria", "real", "fake", "total", "bytes", "tamano"]
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV escrito en {args.csv}")


if __name__ == "__main__":
    main()
