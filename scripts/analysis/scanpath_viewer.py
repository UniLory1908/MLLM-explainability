from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


class ViewerInputError(ValueError):
    pass


class MetadataContractError(ValueError):
    pass


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rgb_array(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def safe_name(value: str, default: str = "tok", max_len: int = 60) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("._-")
    return (cleaned or default)[:max_len]


def detect_metadata_capabilities(metadata: dict) -> dict:
    step_records = metadata.get("step_records")
    if not isinstance(step_records, list) or not step_records:
        return {
            "has_step_records": False,
            "has_heatmap_paths": False,
            "scanpath_ready": False,
            "reason": "missing_or_empty_step_records",
        }

    has_heatmap_paths = any(bool(step.get("heatmap_path")) for step in step_records if isinstance(step, dict))
    has_hotspots = all(isinstance(step, dict) and "hotspots" in step for step in step_records)
    has_dominant = all(isinstance(step, dict) and "dominant_hotspot" in step for step in step_records)
    scanpath = metadata.get("scanpath")
    has_tracks = isinstance(scanpath, dict) and isinstance(scanpath.get("tracks"), list)

    scanpath_ready = bool(has_heatmap_paths and has_hotspots and has_dominant and has_tracks)
    if scanpath_ready:
        reason = "scanpath_ready"
    elif has_heatmap_paths:
        reason = "heatmap_only"
    else:
        reason = "missing_heatmap_paths"

    return {
        "has_step_records": True,
        "has_heatmap_paths": bool(has_heatmap_paths),
        "has_hotspots": bool(has_hotspots),
        "has_dominant_hotspot": bool(has_dominant),
        "has_scanpath_tracks": bool(has_tracks),
        "scanpath_ready": bool(scanpath_ready),
        "reason": reason,
    }


def validate_cli_args(args: argparse.Namespace) -> None:
    metadata_path = Path(args.metadata)
    if not metadata_path.exists() or not metadata_path.is_file():
        raise ViewerInputError(f"--metadata non trovato: {metadata_path}")
    if args.sheet_cols < 1:
        raise ViewerInputError("--sheet-cols deve essere >= 1")
    if args.gif_ms < 1:
        raise ViewerInputError("--gif-ms deve essere >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise ViewerInputError("--max-steps deve essere >= 1 se specificato")
    if args.dominant_tail < 2:
        raise ViewerInputError("--dominant-tail deve essere >= 2")


def validate_metadata_contract(metadata: dict, allow_heatmap_only: bool) -> tuple[str, dict]:
    caps = detect_metadata_capabilities(metadata)
    if not caps["has_step_records"]:
        raise MetadataContractError("Metadata non valido: step_records mancante o vuoto.")

    if caps["scanpath_ready"]:
        return "scanpath", caps

    if allow_heatmap_only and caps["has_heatmap_paths"]:
        return "heatmap_only", caps

    raise MetadataContractError(
        "Metadata non compatibile con la vista scanpath. Servono: step_records[].heatmap_path, "
        "step_records[].hotspots, step_records[].dominant_hotspot, scanpath.tracks. "
        "Riesegui il runner aggiornato oppure usa --allow-heatmap-only per i metadata legacy."
    )


def draw_hotspots_and_tracks(
    heatmap: np.ndarray,
    hotspots: list[dict],
    active_tracks: list[dict],
    dominant_path: list[dict],
    token_label: str = "",
    show_secondary_tracks: bool = False,
    dominant_tail: int = 24,
) -> Image.Image:
    image = Image.fromarray(heatmap, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    for hotspot in hotspots:
        x = float(hotspot["centroid_x"])
        y = float(hotspot["centroid_y"])
        area = max(1.0, float(hotspot.get("area", 1)))
        radius = float(np.clip(np.sqrt(area) * 0.8, 4.0, 20.0))
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=(255, 255, 0, 255), width=2)
        dot_r = 3.2
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=(255, 255, 255, 255))

    if show_secondary_tracks:
        for track in active_tracks:
            points = track.get("points", [])
            if len(points) < 2:
                continue
            xy = [(float(p["centroid_x"]), float(p["centroid_y"])) for p in points]
            draw.line(xy, fill=(70, 220, 255, 110), width=1)

    if len(dominant_path) >= 2:
        prev_point = dominant_path[-2]
        curr_point = dominant_path[-1]
        dominant_xy = [
            (float(prev_point["centroid_x"]), float(prev_point["centroid_y"])),
            (float(curr_point["centroid_x"]), float(curr_point["centroid_y"])),
        ]
        draw.line(dominant_xy, fill=(255, 255, 255, 225), width=3)
        for idx, p in enumerate([prev_point, curr_point]):
            x = float(p["centroid_x"])
            y = float(p["centroid_y"])
            r = 4.5 if idx == 1 else 3.0
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255), outline=(0, 0, 0, 220), width=1)
    elif len(dominant_path) == 1:
        p = dominant_path[-1]
        x = float(p["centroid_x"])
        y = float(p["centroid_y"])
        r = 4.5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255), outline=(0, 0, 0, 220), width=1)

    if token_label:
        label = f"token: {token_label}"
        bbox = draw.textbbox((12, 10), label)
        pad_x = 8
        pad_y = 6
        bg = [
            bbox[0] - pad_x,
            bbox[1] - pad_y,
            bbox[2] + pad_x,
            bbox[3] + pad_y,
        ]
        draw.rounded_rectangle(bg, radius=6, fill=(0, 0, 0, 170), outline=(255, 255, 255, 130), width=1)
        draw.text((12, 10), label, fill=(255, 255, 255, 255))

    return image


def active_tracks_until_step(tracks: list[dict], step_idx: int) -> list[dict]:
    active = []
    for track in tracks:
        prefix = [p for p in track.get("points", []) if int(p.get("step_idx", -1)) <= step_idx]
        if not prefix:
            continue
        active.append({"track_id": track.get("track_id", -1), "points": prefix})
    return active


def dominant_path_until_step(step_records: list[dict], step_idx: int) -> list[dict]:
    points = []
    for step in step_records:
        idx = int(step.get("step_idx", -1))
        if idx > step_idx:
            continue
        hotspot = step.get("dominant_hotspot")
        if not hotspot:
            continue
        points.append({
            "step_idx": idx,
            "centroid_x": float(hotspot["centroid_x"]),
            "centroid_y": float(hotspot["centroid_y"]),
        })
    return points


def build_scanpath_frames_with_stats(
    metadata: dict,
    output_dir: Path,
    max_steps: int | None = None,
    show_secondary_tracks: bool = False,
    dominant_tail: int = 24,
) -> tuple[list[Path], dict]:
    ensure_dir(output_dir)
    step_records = metadata.get("step_records", [])
    tracks = metadata.get("scanpath", {}).get("tracks", [])
    frames: list[Path] = []
    skipped_missing_heatmap_path = 0
    skipped_missing_heatmap_file = 0
    if max_steps is not None:
        step_records = step_records[: max(0, int(max_steps))]

    for step in step_records:
        step_idx = int(step.get("step_idx", len(frames)))
        heatmap_path = step.get("heatmap_path")
        if not heatmap_path:
            skipped_missing_heatmap_path += 1
            continue
        if not Path(heatmap_path).exists():
            skipped_missing_heatmap_file += 1
            continue
        heatmap = rgb_array(heatmap_path)
        hotspots = step.get("hotspots", []) or []
        active_tracks = active_tracks_until_step(tracks, step_idx)
        dominant_path = dominant_path_until_step(step_records, step_idx)
        frame = draw_hotspots_and_tracks(
            heatmap,
            hotspots,
            active_tracks,
            dominant_path,
            token_label=str(step.get("token_label", "")),
            show_secondary_tracks=show_secondary_tracks,
            dominant_tail=dominant_tail,
        )

        token = safe_name(step.get("token_label", "tok"))
        out_path = output_dir / f"step_{step_idx:04d}_{token}.png"
        frame.save(out_path)
        frames.append(out_path)
    stats = {
        "steps_considered": len(step_records),
        "frames_generated": len(frames),
        "skipped_missing_heatmap_path": skipped_missing_heatmap_path,
        "skipped_missing_heatmap_file": skipped_missing_heatmap_file,
    }
    return frames, stats


def build_scanpath_frames(
    metadata: dict,
    output_dir: Path,
    max_steps: int | None = None,
    show_secondary_tracks: bool = False,
    dominant_tail: int = 24,
) -> list[Path]:
    frames, _ = build_scanpath_frames_with_stats(
        metadata=metadata,
        output_dir=output_dir,
        max_steps=max_steps,
        show_secondary_tracks=show_secondary_tracks,
        dominant_tail=dominant_tail,
    )
    return frames


def save_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int = 220) -> None:
    if not frame_paths:
        return
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def save_contact_sheet(frame_paths: list[Path], output_path: Path, cols: int = 8) -> None:
    if not frame_paths:
        return
    if cols < 1:
        raise ViewerInputError("save_contact_sheet: cols deve essere >= 1")
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    rows = int(np.ceil(len(images) / max(1, cols)))
    pad = 6
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), (0, 0, 0))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = pad + col * (w + pad)
        y = pad + row * (h + pad)
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        canvas.paste(img, (x, y))
    canvas.save(output_path)


def quick_preview(frame_paths: list[Path], n: int = 12) -> None:
    if not frame_paths:
        return
    sample = frame_paths[:n]
    cols = min(4, len(sample))
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[len(sample):]:
        ax.axis("off")
    for ax, path in zip(axes, sample):
        ax.imshow(rgb_array(path))
        ax.set_title(path.stem[:28], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dei frame scanpath a partire da metadata.json")
    parser.add_argument("--metadata", required=True, help="Percorso di metadata.json")
    parser.add_argument("--out-dir", help="Cartella di output. Default: <prompt_dir>/scanpath_views")
    parser.add_argument("--max-steps", type=int, help="Numero massimo di step da renderizzare.")
    parser.add_argument("--gif-ms", type=int, default=220, help="Durata dei frame GIF in millisecondi.")
    parser.add_argument("--sheet-cols", type=int, default=8, help="Numero di colonne della contact sheet.")
    parser.add_argument("--allow-heatmap-only", action="store_true", help="Permette metadata legacy senza campi scanpath.")
    parser.add_argument("--show-secondary-tracks", action="store_true", help="Mostra anche le tracce secondarie.")
    parser.add_argument("--dominant-tail", type=int, default=24, help="Parametro mantenuto per compatibilita' con le chiamate esistenti.")
    parser.add_argument("--no-preview", action="store_true", help="Disattiva l'anteprima matplotlib.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validate_cli_args(args)
        metadata_path = Path(args.metadata).resolve()
        metadata = load_json(metadata_path)
        mode, caps = validate_metadata_contract(metadata, allow_heatmap_only=args.allow_heatmap_only)

        default_folder = "scanpath_views" if mode == "scanpath" else "heatmap_views"
        default_out = metadata_path.parent / default_folder
        out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out
        ensure_dir(out_dir)

        frame_paths, stats = build_scanpath_frames_with_stats(
            metadata=metadata,
            output_dir=out_dir,
            max_steps=args.max_steps,
            show_secondary_tracks=args.show_secondary_tracks if mode == "scanpath" else False,
            dominant_tail=args.dominant_tail,
        )

        if not frame_paths:
            print("stato: fallito")
            print(f"mode: {mode}")
            print("motivo: nessun frame valido")
            print(f"step_considerati: {stats['steps_considered']}")
            print(f"step_saltati_senza_heatmap_path: {stats['skipped_missing_heatmap_path']}")
            print(f"step_saltati_heatmap_mancante: {stats['skipped_missing_heatmap_file']}")
            raise SystemExit(2)

        gif_path = out_dir / "scanpath.gif"
        sheet_path = out_dir / "scanpath_contact_sheet.jpg"
        save_gif(frame_paths, gif_path, duration_ms=args.gif_ms)
        save_contact_sheet(frame_paths, sheet_path, cols=args.sheet_cols)

        skips = stats["skipped_missing_heatmap_path"] + stats["skipped_missing_heatmap_file"]
        status = "completato" if skips == 0 else "completato_con_scarti"
        print(f"stato: {status}")
        print(f"mode: {mode}")
        print(f"capacita_metadata: {json.dumps(caps, ensure_ascii=True)}")
        print(f"frame_generati: {stats['frames_generated']}")
        print(f"step_considerati: {stats['steps_considered']}")
        print(f"step_saltati_senza_heatmap_path: {stats['skipped_missing_heatmap_path']}")
        print(f"step_saltati_heatmap_mancante: {stats['skipped_missing_heatmap_file']}")
        print(f"cartella: {out_dir}")
        if gif_path.exists():
            print(f"gif_salvata: {gif_path}")
        if sheet_path.exists():
            print(f"contact_sheet_salvata: {sheet_path}")

        if not args.no_preview:
            quick_preview(frame_paths)
    except ViewerInputError as err:
        print("stato: fallito")
        print("motivo: input_non_valido")
        print(f"dettaglio: {err}")
        raise SystemExit(2) from err
    except (json.JSONDecodeError, OSError) as err:
        print("stato: fallito")
        print("motivo: metadata_non_legibile")
        print(f"dettaglio: {err}")
        raise SystemExit(2) from err
    except MetadataContractError as err:
        print("stato: fallito")
        print("motivo: contratto_metadata")
        print(f"dettaglio: {err}")
        raise SystemExit(2) from err


if __name__ == "__main__":
    main()
