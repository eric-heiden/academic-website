"""Export visual-calibration trial evidence into data/v3 with credential redaction."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = Path("/home/horde/artifacts/newton-live-mcp-v3")
DATA = HERE / "data" / "v3"
ASSETS = HERE / "assets" / "v3"
TOKEN = re.compile(r'(\\?"token\\?"\s*:\s*\\?")[^"\\]+')
PUBLIC_FILES = ("summary.json", "verification.json", "params.json", "TASK.md", "spec.json", "candidates.jsonl", "anytime.json", "agent.times.jsonl")


def redact(text: str) -> str:
    """Remove loopback authentication tokens that agents may have printed."""
    return TOKEN.sub(r"\1<redacted>", text)


def export_trials(source: Path, name: str) -> dict:
    target = DATA / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    manifest = {}
    archive_path = DATA / f"{name}-transcripts.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for workspace in sorted(p for p in source.iterdir() if p.is_dir() and (p / "spec.json").exists()):
            out = target / workspace.name
            out.mkdir()
            for filename in PUBLIC_FILES:
                path = workspace / filename
                if not path.exists():
                    continue
                text = redact(path.read_text(errors="replace"))
                if filename == "candidates.jsonl":
                    # Candidate logs can hold hundreds of thousands of records; store them compressed.
                    (out / (filename + ".gz")).write_bytes(gzip.compress(text.encode(), mtime=0))
                else:
                    (out / filename).write_text(text)
            for filename in ("agent.jsonl", "agent.stderr", "app.log"):
                path = workspace / filename
                if path.exists():
                    archive.writestr(f"{workspace.name}/{filename}", redact(path.read_text(errors="replace")))
            manifest[workspace.name] = {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())
            }
    manifest["_archive"] = {archive_path.name: hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    (DATA / f"{name}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def pairs(analysis: dict) -> list[dict]:
    index = {}
    for row in analysis["trials"]:
        key = (row["task"], row["model"], row["replicate"])
        index.setdefault(key, {"task": row["task"], "model": row["model"], "replicate": row["replicate"]})[row["condition"]] = row
    order = {"arm_offsets": 0, "cloth_drape": 1, "push": 2}
    models = {"opus": 0, "astra": 1}
    return sorted(index.values(), key=lambda p: (order[p["task"]], models[p["model"]], str(p["replicate"])))


def copy_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True)
    for path in (STUDY / "report-assets").glob("*"):
        shutil.copyfile(path, ASSETS / path.name)
    from PIL import Image  # noqa: PLC0415

    for task in ("arm_offsets", "cloth_drape", "push"):
        reference = Path.home() / ".newton-visual-private" / "reference" / task
        for sheet in reference.glob("sheet_*.png"):
            # Display copies only; the original PNG photos are in the reference archive.
            Image.open(sheet).convert("RGB").save(ASSETS / f"{task}-{sheet.stem}.jpg", quality=85)
        for view in ASSETS.glob(f"{task}-views.png"):
            # The response stacks four 240-pixel rows with 4-pixel gaps; re-tile them 2 x 2 for display.
            strip = Image.open(view).convert("RGB")
            tiles = [strip.crop((0, i * 244, 320, i * 244 + 240)) for i in range(4)]
            grid = Image.new("RGB", (644, 484), (255, 255, 255))
            for i, tile in enumerate(tiles):
                grid.paste(tile, ((i % 2) * 324, (i // 2) * 244))
            grid.save(ASSETS / f"{task}-views.jpg", quality=90)
            view.unlink()
    with zipfile.ZipFile(DATA / "reference-photos.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for task in ("arm_offsets", "cloth_drape", "push"):
            for photo in sorted((Path.home() / ".newton-visual-private" / "reference" / task).iterdir()):
                archive.write(photo, f"{task}/{photo.name}")
        shutil.copyfile(reference / "reference.json", DATA / f"{task}-reference.json")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    copy_assets()
    for name in ("confirmation", "development", "followup"):
        export_trials(STUDY / name, name)
    for filename in ("REGISTRATION.json", "FOLLOWUP_REGISTRATION.json", "DEVELOPMENT_LOG.md", "tests-frozen.log"):
        shutil.copyfile(STUDY / filename, DATA / filename)
    # The study is complete, so the hidden generating parameters are disclosed with the evidence.
    shutil.copyfile(Path.home() / ".newton-visual-private" / "truth.json", DATA / "truth.json")
    print("exported")
