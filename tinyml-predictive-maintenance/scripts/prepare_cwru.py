"""Prepare the CWRU dataset folder.

The CWRU bearing dataset is hosted at:
    https://engineering.case.edu/bearingdatacenter

Server availability is unreliable, especially from outside North America, so
this script supports three workflows:

1. ``--from-url URL --label LABEL`` downloads a single ``.mat`` file into the
   matching label folder. Run it once per file. Recommended when the official
   site is reachable.

2. ``--manifest`` prints the canonical file list with the official URLs so the
   user can download manually with a browser or any HTTP client.

3. ``--synthetic`` writes ``.mat`` files with CWRU-shape arrays, generated from
   the project's own simulator. Output reports must clearly mark these as
   synthetic. This mode lets reviewers run the comparison pipeline without
   waiting on a download, and it doubles as a smoke test for the loader.

Files end up under ``data/cwru/<label>/<file>.mat``. Labels follow the project
convention used by ``tpm.datasets.cwru``: ``normal``, ``inner``, ``outer``,
``ball``.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Keep imports flexible so this script also works without ``pip install -e .``.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

# Canonical CWRU files used by most papers. The URLs are stable IDs from the
# official site. We only list a small representative subset so the comparison
# stays reproducible without forcing a multi-GB download.
MANIFEST: dict[str, list[tuple[str, str]]] = {
    "normal": [
        ("97.mat", "https://engineering.case.edu/sites/default/files/97.mat"),
        ("98.mat", "https://engineering.case.edu/sites/default/files/98.mat"),
    ],
    "inner": [
        ("105.mat", "https://engineering.case.edu/sites/default/files/105.mat"),
        ("106.mat", "https://engineering.case.edu/sites/default/files/106.mat"),
    ],
    "outer": [
        ("130.mat", "https://engineering.case.edu/sites/default/files/130.mat"),
        ("131.mat", "https://engineering.case.edu/sites/default/files/131.mat"),
    ],
    "ball": [
        ("118.mat", "https://engineering.case.edu/sites/default/files/118.mat"),
        ("119.mat", "https://engineering.case.edu/sites/default/files/119.mat"),
    ],
}


def cmd_manifest() -> None:
    """Print the canonical file list and stop."""

    print("CWRU subset used by this project:")
    print()
    for label, files in MANIFEST.items():
        print(f"  data/cwru/{label}/")
        for name, url in files:
            print(f"    {name:>10}  {url}")
        print()
    print("Drop each .mat into its label folder, or use --from-url to download.")


def cmd_from_url(url: str, label: str, dest_root: Path) -> None:
    """Download one file into ``data/cwru/<label>/``."""

    if label not in MANIFEST:
        raise SystemExit(
            f"unknown label '{label}'. Use one of: {sorted(MANIFEST)}"
        )
    folder = dest_root / label
    folder.mkdir(parents=True, exist_ok=True)
    name = Path(url).name or "downloaded.mat"
    out = folder / name
    print(f"downloading {url} -> {out}")
    # ``urlretrieve`` is fine for a one-off CLI; we never call it in a loop with
    # untrusted input. If the server is slow, the user can Ctrl+C and retry.
    urllib.request.urlretrieve(url, out)
    print(f"saved {out.stat().st_size} bytes")


def _download_one(url: str, dest: Path, timeout: float) -> int:
    """Stream one URL to disk with a hard timeout. Returns bytes written.

    ``urlretrieve`` does not honor a timeout, so we open the request manually
    and copy in chunks. Partial files are removed on failure so a retry starts
    clean instead of resuming a corrupt download.
    """

    request = urllib.request.Request(url, headers={"User-Agent": "tpm-prepare-cwru/1.0"})
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as fh:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
        return written
    except Exception:
        if dest.exists():
            dest.unlink()
        raise


def cmd_download_all(dest_root: Path, timeout: float = 60.0, retries: int = 2) -> None:
    """Download every file in MANIFEST. Skip ones already on disk.

    The CWRU server is occasionally slow or unreachable from outside North
    America, so each file gets a few retries with a per-attempt timeout. If a
    file still fails after retries, we keep going and report the failures at
    the end so the user only needs to manually grab the ones that broke.
    """

    failures: list[tuple[str, str, str]] = []
    total_bytes = 0
    for label, files in MANIFEST.items():
        folder = dest_root / label
        folder.mkdir(parents=True, exist_ok=True)
        for name, url in files:
            out = folder / name
            if out.exists() and out.stat().st_size > 0:
                size = out.stat().st_size
                total_bytes += size
                print(f"skip  {label}/{name:<10} ({size} bytes, already present)")
                continue

            attempt = 0
            while True:
                attempt += 1
                try:
                    print(f"get   {label}/{name:<10} <- {url}  (attempt {attempt})")
                    written = _download_one(url, out, timeout=timeout)
                    print(f"  ok  {written} bytes")
                    total_bytes += written
                    break
                except Exception as exc:
                    print(f"  fail {type(exc).__name__}: {exc}")
                    if attempt > retries:
                        failures.append((label, name, str(exc)))
                        break

    print()
    print(f"finished. {total_bytes} bytes written across {dest_root}/")
    if failures:
        print("the following files did not download — get them manually in a browser:")
        for label, name, reason in failures:
            url = next(u for n, u in MANIFEST[label] if n == name)
            print(f"  {label}/{name}  <- {url}  ({reason})")
        sys.exit(1)


def _synthetic_signal(label: str, n_samples: int, seed: int) -> np.ndarray:
    """Return a CWRU-flavored vibration trace for a fault label.

    The shapes loosely mirror the simulator in ``tpm.signal_sim``: a running
    frequency plus noise for normal, plus a label-specific perturbation. These
    values do not claim to match real CWRU statistics; they exist only so the
    pipeline can be smoke-tested end to end.
    """

    rng = np.random.default_rng(seed)
    sample_rate = 12_000
    t = np.arange(n_samples) / sample_rate

    # 30 Hz shaft running frequency, very small second harmonic, plus white
    # noise. The amplitudes are tuned so kurtosis/RMS shifts between labels are
    # detectable without being trivial.
    base = 0.6 * np.sin(2 * np.pi * 30.0 * t)
    harmonic = 0.08 * np.sin(2 * np.pi * 60.0 * t)
    noise = rng.normal(0.0, 0.05, size=n_samples)
    signal = base + harmonic + noise

    if label == "normal":
        pass
    elif label == "inner":
        # Inner-race faults produce strong impulses at the ball-pass frequency.
        signal = signal + 0.4 * np.sin(2 * np.pi * 162.0 * t) * np.sign(
            np.sin(2 * np.pi * 5.0 * t)
        )
    elif label == "outer":
        # Outer-race faults have lower repetition rate but stronger impacts.
        impacts = (rng.random(n_samples) < 0.01).astype(np.float32)
        signal = signal + impacts * rng.normal(1.5, 0.3, size=n_samples)
    elif label == "ball":
        # Ball faults inject modulated mid-band energy.
        signal = signal + 0.3 * np.sin(2 * np.pi * 235.0 * t) * (
            1.0 + 0.5 * np.sin(2 * np.pi * 7.0 * t)
        )
    else:
        raise ValueError(f"unsupported label: {label}")

    return signal.astype(np.float64)


def cmd_synthetic(dest_root: Path, seconds: float = 10.0) -> None:
    """Write CWRU-shape ``.mat`` files generated from the simulator.

    Each file contains a key named ``X{nnn}_DE_time`` so the loader's suffix
    matching picks it up exactly like a real CWRU file would.
    """

    from scipy.io import savemat

    sample_rate = 12_000
    n_samples = int(seconds * sample_rate)

    for label, files in MANIFEST.items():
        folder = dest_root / label
        folder.mkdir(parents=True, exist_ok=True)
        for index, (name, _url) in enumerate(files):
            stem = Path(name).stem  # "97" from "97.mat"
            signal = _synthetic_signal(label, n_samples, seed=hash((label, index)) & 0xFFFF)
            out = folder / name
            # The CWRU naming uses an "X" prefix; we mirror it so the loader
            # works without any special-casing.
            payload = {f"X{stem}_DE_time": signal.reshape(-1, 1)}
            savemat(str(out), payload, do_compression=True)
            print(f"wrote {out} ({signal.size} samples)")
    marker = dest_root / "SYNTHETIC.txt"
    marker.write_text(
        "These .mat files were generated by scripts/prepare_cwru.py --synthetic.\n"
        "Replace them with real CWRU files before quoting numbers as real CWRU results.\n",
        encoding="utf-8",
    )
    print(f"wrote marker: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the CWRU bearing dataset.")
    parser.add_argument(
        "--data-root",
        default="data/cwru",
        help="Destination folder. Defaults to data/cwru.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest", help="Print canonical file list and URLs.")

    p_url = sub.add_parser("from-url", help="Download one .mat into a label folder.")
    p_url.add_argument("url")
    p_url.add_argument("--label", required=True, choices=sorted(MANIFEST))

    p_all = sub.add_parser("download-all", help="Download every file in the manifest.")
    p_all.add_argument("--timeout", type=float, default=60.0, help="Per-attempt timeout in seconds.")
    p_all.add_argument("--retries", type=int, default=2, help="Extra retries per file after the first attempt.")

    p_syn = sub.add_parser("synthetic", help="Generate CWRU-shape .mat files locally.")
    p_syn.add_argument("--seconds", type=float, default=10.0)

    args = parser.parse_args()
    dest_root = Path(args.data_root)

    if args.cmd == "manifest":
        cmd_manifest()
    elif args.cmd == "from-url":
        cmd_from_url(args.url, args.label, dest_root)
    elif args.cmd == "download-all":
        cmd_download_all(dest_root, timeout=args.timeout, retries=args.retries)
    elif args.cmd == "synthetic":
        cmd_synthetic(dest_root, seconds=args.seconds)


if __name__ == "__main__":
    main()
