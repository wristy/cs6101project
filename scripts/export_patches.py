"""Export generated git diffs after reviewing upstream compatibility changes."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "thought-tracing": "966eb2712714aa29b4d5d2395e39af94a2be90d7",
    "AutoToM": "3f569b7ab1d0ee2702ab43ebdc724679d5fc5231",
}


def main():
    destination = ROOT / "patches"
    destination.mkdir(exist_ok=True)
    for name, revision in REVISIONS.items():
        checkout = ROOT / "external" / name
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Refusing to export {name}: unexpected revision {actual}")
        patch = subprocess.check_output(["git", "-C", str(checkout), "diff", "--binary", "HEAD", "--"])
        if not patch:
            raise RuntimeError(f"No compatibility changes found in {name}")
        (destination / f"{name}.patch").write_bytes(patch)
        print(f"Exported {name}: {len(patch)} bytes")


if __name__ == "__main__":
    main()