"""Download, verify and install dataset shards into the canonical local layout."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmwave_densepose.data.download import DownloadCancelled, install_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/release-manifest.json")
    parser.add_argument("--output", default="data/mmwave_densepose")
    parser.add_argument("--cache-dir", default="data/downloads")
    parser.add_argument("--archive-dir", help="Use already downloaded archives instead of the network")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--archive-id", action="append", dest="archive_ids",
                        help="Install only this mapped archive ID; repeat for more parts")
    parser.add_argument("--repair", action="store_true", help="Replace corrupt files in the selected release parts")
    parser.add_argument("--progress-json", action="store_true", help="Write structured progress to stderr")
    args = parser.parse_args()
    try:
        print(json.dumps(install_dataset(args.manifest, args.output, args.cache_dir,
                                        args.archive_dir, args.dry_run,
                                        archive_ids=args.archive_ids, repair=args.repair,
                                        progress=(lambda update: print(json.dumps(update), file=sys.stderr, flush=True))
                                        if args.progress_json else None), indent=2))
    except (ValueError, OSError, KeyError, DownloadCancelled) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
