"""Download, verify and install official paper weights or prediction packages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmwave_densepose.data.download import DownloadCancelled
from mmwave_densepose.data.paper_assets import PaperAssetBusy, PaperAssetInstaller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="paper/download-manifest.json", help="Repository-relative package manifest")
    parser.add_argument("--kind", choices=("weights", "predictions", "all"), default="weights")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--asset", action="append", dest="assets", help="Official asset ID; repeat to select several")
    selection.add_argument("--all", action="store_true", help="Select every registered asset of the chosen kind")
    parser.add_argument("--test-id", help="Select one official test ID when installing predictions")
    parser.add_argument("--list", action="store_true", help="List availability without downloading or hashing files")
    parser.add_argument("--verify-only", action="store_true", help="Check installed files against pinned SHA256 values")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected package plan without writing files")
    parser.add_argument("--archive-dir", help="Install already downloaded ZIPs from this directory")
    parser.add_argument("--cache-dir", help="Downloaded ZIP cache; defaults to data/downloads/paper-assets")
    parser.add_argument("--repair", action="store_true", help="Explicitly replace files which differ from the official package")
    parser.add_argument("--progress-json", action="store_true", help="Write structured progress to stderr")
    args = parser.parse_args(argv)
    if not (args.assets or args.all or args.list):
        parser.error("select --asset ID or --all, or use --list to inspect available assets")
    if args.test_id and args.kind == "weights":
        parser.error("--test-id applies only to predictions; select --kind predictions")
    try:
        installer = PaperAssetInstaller(ROOT, manifest_path=args.manifest, cache_dir=args.cache_dir)
        inventory = installer.inventory(kind=None if args.kind == "all" else args.kind, test_id=args.test_id)
        items = inventory["items"]
        if args.assets:
            unknown = set(args.assets) - {item["asset_id"] for item in items}
            if unknown:
                parser.error("unknown asset selection: " + ", ".join(sorted(unknown)))
            items = [item for item in items if item["asset_id"] in args.assets]
        if args.list:
            inventory["items"] = items
            print(json.dumps(inventory, indent=2))
            return 0
        # Unpublished predictions are deliberately absent, rather than being
        # silently replaced by another test. --all selects published packages.
        package_ids = [item["package_id"] for item in items
                       if item["filename"] is not None or (args.verify_only and item["kind"] == "weights")]
        if args.assets and len(package_ids) != len(items):
            raise ValueError("No package is published for one or more selected official tests")
        if not package_ids:
            raise ValueError("No package is published for this selection")
        progress = (lambda update: print(json.dumps(update), file=sys.stderr, flush=True)) if args.progress_json else None
        if args.verify_only:
            result = installer.verify(package_ids, progress=progress)
        else:
            result = installer.install(package_ids, archive_dir=args.archive_dir, repair=args.repair,
                                       dry_run=args.dry_run, progress=progress)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] != "invalid" else 1
    except (ValueError, OSError, KeyError, PaperAssetBusy, DownloadCancelled) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
