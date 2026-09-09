"""Freeze pilot IDs and load source-verified notebook inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from tom_research.data import load_autotom_examples, load_examples, select_pilot


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "external/thought-tracing/data/paraphrased_tomi/test.txt"
MANIFEST = ROOT / "data/thoughttracing_pilot_ids.json"
TRACKS = {
    "thoughttracing": {"source": SOURCE, "dataset": "ParaphrasedToMi", "answer_format": "free_text"},
    "autotom": {"source": ROOT / "external/AutoToM/benchmarks/data/ToMi/test_balanced.jsonl",
                "dataset": "AutoToM-ToMi", "answer_format": "multiple_choice"},
}


def track_paths(track):
    return TRACKS[track]["source"], ROOT / "data" / f"{track}_pilot_ids.json"


def track_examples(source, track):
    return load_autotom_examples(source) if track == "autotom" else load_examples(source)


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def prepare_pilot(source=SOURCE, manifest_path=MANIFEST, *, track="thoughttracing"):
    source, manifest_path = Path(source), Path(manifest_path)
    if manifest_path.exists():
        load_study(source, manifest_path, track=track)
        return
    settings = TRACKS[track]
    examples = track_examples(source, track)
    debug, pilot = select_pilot(examples)
    manifest = {
        "track": track, "methods": ["plain", track],
        "dataset": settings["dataset"], "split": "test", "answer_format": settings["answer_format"],
        "seed": 42, "source_sha256": file_hash(source),
        "eligible_first_order_questions": len(examples),
        "debug_ids": [item.id for item in debug], "pilot_ids": [item.id for item in pilot],
        "selection": "10 first-order true and 10 false belief questions; distinct story texts",
        "comparison": "Small first-order pilot, not comparable to full published benchmark scores",
    }
    if track == "thoughttracing":
        manifest["trace_sha256"] = file_hash(source.with_suffix(".trace"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def load_study(source=SOURCE, manifest_path=MANIFEST, *, track="thoughttracing"):
    source = Path(source)
    metadata = json.loads(Path(manifest_path).read_text())
    if metadata.get("track", track) != track:
        raise ValueError("Manifest belongs to another track")
    if (metadata["source_sha256"] != file_hash(source)
            or (track == "thoughttracing" and metadata["trace_sha256"] != file_hash(source.with_suffix(".trace")))):
        raise ValueError("Dataset hash differs from frozen pilot manifest")
    if metadata.get("track") != track or metadata.get("methods") != ["plain", track]:
        raise ValueError("Legacy manifest lacks track configuration; prepare a new track manifest")
    examples = {item.id: item for item in track_examples(source, track)}
    debug = [examples[item_id] for item_id in metadata["debug_ids"]]
    pilot = [examples[item_id] for item_id in metadata["pilot_ids"]]
    if (len(debug) != 2 or len(pilot) != 20
            or len({item.story_id for item in debug + pilot}) != 22):
        raise ValueError("Frozen pilot has invalid size or overlapping stories")
    return debug, pilot, metadata


def provenance(root=ROOT, *, track="thoughttracing"):
    root = Path(root)
    sources = {}
    for name in ("thought-tracing", "AutoToM"):
        checkout = root / "external" / name
        commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(checkout), "diff", "HEAD", "--"])
        sources[name] = {"commit": commit, "patch_sha256": sha256(diff).hexdigest()}
    sources["helpers"] = {str(path.relative_to(root)): file_hash(path)
                          for path in sorted((root / "src/tom_research").glob("*.py"))}
    sources["pilot_ids_sha256"] = file_hash(root / "data" / f"{track}_pilot_ids.json")
    lock = root / "uv.lock"
    if lock.exists():
        sources["environment_lock_sha256"] = file_hash(lock)
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--track", choices=list(TRACKS))
    args = parser.parse_args()
    for track in [args.track] if args.track else TRACKS:
        source, manifest = track_paths(track)
        prepare_pilot(source, manifest, track=track)
        debug, pilot, metadata = load_study(source, manifest, track=track)
        print(f"{track}: {len(debug)} debugging and {len(pilot)} pilot questions.")
        print(f"Source: {metadata['source_sha256']}")


if __name__ == "__main__":
    main()