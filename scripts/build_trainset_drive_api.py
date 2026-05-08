"""Construye el trainset de ejecucion via Drive API con streaming.

Evita el problema de la cache FUSE de Google Drive en Colab. En vez de leer
el zip a traves del mount de Drive (que descarga ~3.5 GB de cache local por
cada categoria a la que se accede), aqui:

1. Se autentica con la API REST de Drive.
2. Resuelve el shortcut al fichero original si hace falta.
3. Descarga la "central directory" del zip con un solo range request (pocos
   MB al final del fichero).
4. Muestrea N imagenes por categoria con seed determinista.
5. Abre un streaming GET desde el primer offset seleccionado y, recorriendo
   el fichero secuencialmente, escribe a disco solo los ficheros del subset
   y descarta el resto sin guardarlo.

Asi el disco de Colab solo recibe el subset (~7 GB para N=5000) y nunca el
zip entero. Toda la transferencia va directa de los servidores de Drive al
proceso Python; no toca el mount FUSE.

Pre-requisito en Colab: haber llamado antes a auth.authenticate_user().

Uso:
    python scripts/build_trainset_drive_api.py \
        --file-id 1abc...xyz \
        --out /content/cnndetection/progan_train \
        --n-per-cat 5000 \
        --seed 42 \
        --manifest reports/trainset_ejecucion_manifest.json

--file-id admite el id del shortcut directamente; el script resuelve a
shortcutDetails.targetId. Para encontrarlo, usar la celda del notebook que
hace files().list() por nombre.
"""

import argparse
import json
import random
import struct
import time
import zlib
from collections import defaultdict
from pathlib import Path

import google.auth
import google.auth.transport.requests as gar
import requests

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REAL_DIRS = {"0_real", "nature"}
FAKE_DIRS = {"1_fake", "ai"}
DRIVE_API = "https://www.googleapis.com/drive/v3"


def get_creds():
    creds, _ = google.auth.default()
    creds.refresh(gar.Request())
    return creds


def _api_get(creds, path, params=None, headers=None, stream=False, timeout=60):
    h = {"Authorization": f"Bearer {creds.token}"}
    if headers:
        h.update(headers)
    p = {"supportsAllDrives": "true"}
    if params:
        p.update(params)
    return requests.get(
        f"{DRIVE_API}{path}", params=p, headers=h, stream=stream, timeout=timeout
    )


def get_metadata(creds, file_id):
    """Devuelve (target_id, file_size, name); resuelve shortcuts en cadena."""
    while True:
        r = _api_get(
            creds,
            f"/files/{file_id}",
            {"fields": "id,name,size,mimeType,shortcutDetails"},
        )
        r.raise_for_status()
        data = r.json()
        if data.get("mimeType") == "application/vnd.google-apps.shortcut":
            file_id = data["shortcutDetails"]["targetId"]
            continue
        return data["id"], int(data["size"]), data["name"]


def download_range(creds, file_id, start, end):
    r = _api_get(
        creds,
        f"/files/{file_id}",
        {"alt": "media"},
        {"Range": f"bytes={start}-{end}"},
    )
    r.raise_for_status()
    return r.content


def _parse_zip64_extra(extra, want_uncomp, want_comp, want_offset):
    pos = 0
    while pos + 4 <= len(extra):
        ext_id = struct.unpack("<H", extra[pos : pos + 2])[0]
        ext_size = struct.unpack("<H", extra[pos + 2 : pos + 4])[0]
        if ext_id == 0x0001:
            z = pos + 4
            uncomp = comp = offset = None
            if want_uncomp:
                uncomp = struct.unpack("<Q", extra[z : z + 8])[0]
                z += 8
            if want_comp:
                comp = struct.unpack("<Q", extra[z : z + 8])[0]
                z += 8
            if want_offset:
                offset = struct.unpack("<Q", extra[z : z + 8])[0]
                z += 8
            return uncomp, comp, offset
        pos += 4 + ext_size
    return None, None, None


def get_central_directory(creds, file_id, file_size):
    """Lee EOCD (con soporte zip64) y devuelve la lista de file entries."""
    search = min(65 * 1024, file_size)
    tail = download_range(creds, file_id, file_size - search, file_size - 1)

    eocd_idx = tail.rfind(b"PK\x05\x06")
    if eocd_idx < 0:
        raise ValueError("EOCD no encontrada en los ultimos 64 KB")

    eocd = tail[eocd_idx : eocd_idx + 22]
    cd_size = struct.unpack("<I", eocd[12:16])[0]
    cd_offset = struct.unpack("<I", eocd[16:20])[0]
    total_entries = struct.unpack("<H", eocd[10:12])[0]

    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or total_entries == 0xFFFF:
        # zip64
        z64_loc = tail.rfind(b"PK\x06\x07", 0, eocd_idx)
        if z64_loc < 0:
            raise ValueError("Zip64 EOCD locator no encontrado")
        z64_eocd_off = struct.unpack("<Q", tail[z64_loc + 8 : z64_loc + 16])[0]
        z64_eocd = download_range(creds, file_id, z64_eocd_off, z64_eocd_off + 56 - 1)
        if z64_eocd[:4] != b"PK\x06\x06":
            raise ValueError("Firma zip64 EOCD invalida")
        cd_size = struct.unpack("<Q", z64_eocd[40:48])[0]
        cd_offset = struct.unpack("<Q", z64_eocd[48:56])[0]

    print(f"  CD: offset={cd_offset}, size={cd_size / 1024 ** 2:.1f} MB")
    cd = download_range(creds, file_id, cd_offset, cd_offset + cd_size - 1)

    files = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        flags = struct.unpack("<H", cd[pos + 8 : pos + 10])[0]
        method = struct.unpack("<H", cd[pos + 10 : pos + 12])[0]
        comp_size = struct.unpack("<I", cd[pos + 20 : pos + 24])[0]
        uncomp_size = struct.unpack("<I", cd[pos + 24 : pos + 28])[0]
        fname_len = struct.unpack("<H", cd[pos + 28 : pos + 30])[0]
        extra_len = struct.unpack("<H", cd[pos + 30 : pos + 32])[0]
        comment_len = struct.unpack("<H", cd[pos + 32 : pos + 34])[0]
        local_offset = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        filename = cd[pos + 46 : pos + 46 + fname_len].decode("utf-8", errors="replace")
        extra = cd[pos + 46 + fname_len : pos + 46 + fname_len + extra_len]

        if (
            comp_size == 0xFFFFFFFF
            or uncomp_size == 0xFFFFFFFF
            or local_offset == 0xFFFFFFFF
        ):
            zu, zc, zo = _parse_zip64_extra(
                extra,
                uncomp_size == 0xFFFFFFFF,
                comp_size == 0xFFFFFFFF,
                local_offset == 0xFFFFFFFF,
            )
            if zu is not None:
                uncomp_size = zu
            if zc is not None:
                comp_size = zc
            if zo is not None:
                local_offset = zo

        files.append(
            {
                "filename": filename,
                "method": method,
                "flags": flags,
                "comp_size": comp_size,
                "uncomp_size": uncomp_size,
                "offset": local_offset,
            }
        )
        pos += 46 + fname_len + extra_len + comment_len

    return files


def parse_path(filename):
    parts = [p for p in filename.replace("\\", "/").split("/") if p]
    for i, p in enumerate(parts):
        pl = p.lower()
        if pl in REAL_DIRS:
            return (parts[i - 1] if i > 0 else None), "0_real"
        if pl in FAKE_DIRS:
            return (parts[i - 1] if i > 0 else None), "1_fake"
    return None, None


def sample(files, n_per_cat, seed):
    rng = random.Random(seed)
    grouped = defaultdict(lambda: {"0_real": [], "1_fake": []})

    for f in files:
        if Path(f["filename"]).suffix.lower() not in VALID_EXTS:
            continue
        cat, label = parse_path(f["filename"])
        if cat is None:
            continue
        grouped[cat][label].append(f)

    per_label = n_per_cat // 2
    selected = []
    cats = sorted(grouped.keys())

    print(f"\n{'categoria':<20} {'real':>8} {'fake':>8} {'subset':>10}")
    print("-" * 50)
    for cat in cats:
        n_r_total = len(grouped[cat]["0_real"])
        n_f_total = len(grouped[cat]["1_fake"])
        n_r = min(per_label, n_r_total)
        n_f = min(per_label, n_f_total)
        if n_r:
            selected.extend(rng.sample(grouped[cat]["0_real"], n_r))
        if n_f:
            selected.extend(rng.sample(grouped[cat]["1_fake"], n_f))
        print(f"{cat:<20} {n_r:>8} {n_f:>8} {n_r + n_f:>10}")
    print("-" * 50)
    print(f"{'TOTAL':<20} {'':>8} {'':>8} {len(selected):>10}\n")
    return selected


def _read_exact(raw, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = raw.read(n - len(buf))
        if not chunk:
            raise IOError(f"Stream truncado: faltan {n - len(buf)} bytes")
        buf.extend(chunk)
    return bytes(buf)


def _skip_exact(raw, n):
    while n > 0:
        chunk = raw.read(min(n, 4 * 1024 * 1024))
        if not chunk:
            raise IOError(f"Stream truncado al saltar; restantes={n}")
        n -= len(chunk)


def _stream_once(creds, file_id, selected_sorted, out_dir, progress_every):
    """Una pasada de streaming. Lanza IOError si la conexion cae."""
    start_off = selected_sorted[0]["offset"]

    r = _api_get(
        creds,
        f"/files/{file_id}",
        {"alt": "media"},
        {"Range": f"bytes={start_off}-"},
        stream=True,
        timeout=300,
    )
    r.raise_for_status()
    raw = r.raw

    written = 0
    skipped_existing = 0
    bytes_streamed = 0
    bytes_written = 0
    t0 = time.time()
    pos = start_off

    try:
        for idx, target in enumerate(selected_sorted, 1):
            target_off = target["offset"]

            if pos < target_off:
                _skip_exact(raw, target_off - pos)
                bytes_streamed += target_off - pos
                pos = target_off
            elif pos > target_off:
                raise RuntimeError(f"pos {pos} > target {target_off}")

            header = _read_exact(raw, 30)
            pos += 30
            bytes_streamed += 30
            if header[:4] != b"PK\x03\x04":
                raise ValueError(
                    f"Bad local header en offset {target_off}: {header[:4]!r}"
                )
            fname_len = struct.unpack("<H", header[26:28])[0]
            extra_len = struct.unpack("<H", header[28:30])[0]

            _skip_exact(raw, fname_len + extra_len)
            pos += fname_len + extra_len
            bytes_streamed += fname_len + extra_len

            comp_size = target["comp_size"]
            cat, label = parse_path(target["filename"])
            basename = Path(target["filename"]).name
            dest = out_dir / cat / label / basename

            if dest.exists() and dest.stat().st_size > 0:
                _skip_exact(raw, comp_size)
                pos += comp_size
                bytes_streamed += comp_size
                skipped_existing += 1
            else:
                data = _read_exact(raw, comp_size)
                pos += comp_size
                bytes_streamed += comp_size

                method = target["method"]
                if method == 0:
                    payload = data
                elif method == 8:
                    payload = zlib.decompress(data, -15)
                else:
                    print(
                        f"WARN: metodo {method} no soportado, saltando "
                        f"{target['filename']}"
                    )
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
                bytes_written += len(payload)
                written += 1

            if idx % progress_every == 0 or idx == len(selected_sorted):
                el = max(time.time() - t0, 1e-9)
                pct = 100 * idx / len(selected_sorted)
                speed_dl = bytes_streamed / el / 1024 ** 2
                print(
                    f"  [{idx:6d}/{len(selected_sorted)}] {pct:5.1f}%  "
                    f"escritos={written}  ya_existian={skipped_existing}  "
                    f"DL={bytes_streamed / 1024 ** 3:.2f} GB ({speed_dl:.1f} MB/s)  "
                    f"WR={bytes_written / 1024 ** 2:.0f} MB  "
                    f"t={el / 60:.1f} min"
                )
    finally:
        try:
            r.close()
        except Exception:
            pass

    return written, skipped_existing


def stream_extract(creds, file_id, selected, out_dir, max_retries=3, progress_every=500):
    selected_sorted = sorted(selected, key=lambda f: f["offset"])
    if not selected_sorted:
        return 0, 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0
    pending = selected_sorted

    for attempt in range(max_retries + 1):
        if not pending:
            break
        # Filtrar lo ya hecho para reanudar despues de un retry sin re-bajar todo
        remaining = []
        for t in pending:
            cat, label = parse_path(t["filename"])
            dest = out_dir / cat / label / Path(t["filename"]).name
            if dest.exists() and dest.stat().st_size > 0:
                total_skipped += 1
            else:
                remaining.append(t)
        if not remaining:
            break

        try:
            print(
                f"Pasada {attempt + 1}: streaming {len(remaining)} ficheros "
                f"desde offset {remaining[0]['offset']}"
            )
            w, s = _stream_once(creds, file_id, remaining, out_dir, progress_every)
            total_written += w
            total_skipped += s
            pending = []
        except (
            requests.exceptions.RequestException,
            IOError,
            ValueError,
        ) as e:
            print(f"  fallo: {type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  reintentando en {wait}s...")
                time.sleep(wait)
                # Refrescar token por si caduco
                creds.refresh(gar.Request())
                pending = remaining
            else:
                raise

    return total_written, total_skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-id",
        required=True,
        help="ID del progan_train.zip en Drive (admite shortcut).",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-per-cat", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo lee CD y escribe el manifest; no streamea.",
    )
    args = parser.parse_args()

    if args.n_per_cat % 2 != 0:
        raise SystemExit("--n-per-cat debe ser par.")

    print("Autenticando...")
    creds = get_creds()

    print(f"Resolviendo file-id {args.file_id}...")
    target_id, file_size, name = get_metadata(creds, args.file_id)
    print(f"  -> {name} ({file_size / 1024 ** 3:.2f} GB) id={target_id}")

    print("Descargando central directory...")
    files = get_central_directory(creds, target_id, file_size)
    print(f"  -> {len(files)} entradas en el zip")

    print(f"Muestreando (n_per_cat={args.n_per_cat}, seed={args.seed})...")
    selected = sample(files, args.n_per_cat, args.seed)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "file_id": target_id,
            "file_name": name,
            "file_size": file_size,
            "n_per_cat": args.n_per_cat,
            "seed": args.seed,
            "actual_total": len(selected),
            "files": [f["filename"] for f in selected],
        }
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"  manifest -> {args.manifest}")

    if args.dry_run:
        print("[dry-run] no se extrae nada.")
        return

    print(f"\nExtrayendo a {args.out}...")
    written, skipped = stream_extract(creds, target_id, selected, args.out)
    print(f"\nOK: {written} escritas, {skipped} ya existian.")


if __name__ == "__main__":
    main()
