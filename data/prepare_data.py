"""Rebuild the train/test image folders used by the notebooks.

The project used a fixed split of the NIH/NLM malaria cell images: 1,300
images per class for testing and the remaining 12,479 per class for
training. test_split.csv (next to this script) lists the 2,600 test images.

This script copies the images from the unzipped NIH download into the
layout the notebooks expect:

    DEST/train/parasitized    DEST/train/uninfected
    DEST/test/parasitized     DEST/test/uninfected

Usage:
    python data/prepare_data.py --source cell_images --dest data/cell_images

--source is the folder unzipped from cell_images.zip; it contains
Parasitized/ and Uninfected/.

Added in 2026 when the project was published; not part of the original
2023 notebooks.
"""
import argparse
import csv
import os
import shutil
import sys

# Output folder name -> folder name inside the NIH download.
CLASS_DIRS = {"parasitized": "Parasitized", "uninfected": "Uninfected"}
EXPECTED_PER_CLASS = {"train": 12479, "test": 1300}
DEFAULT_SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_split.csv")


def read_test_split(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {(row["class"], row["filename"]) for row in csv.DictReader(fh)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="unzipped cell_images folder from NIH")
    parser.add_argument("--dest", required=True, help="output folder for train/ and test/")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="CSV of test images (default: data/test_split.csv)")
    args = parser.parse_args()

    test = read_test_split(args.split)
    counts = {}
    for cls, nih_dir in CLASS_DIRS.items():
        src_dir = os.path.join(args.source, nih_dir)
        if not os.path.isdir(src_dir):
            sys.exit(f"Folder not found: {src_dir}\nPoint --source at the unzipped cell_images folder.")
        # The NIH folders also hold a Thumbs.db file; only the PNG images are used.
        names = sorted(n for n in os.listdir(src_dir) if n.lower().endswith(".png"))
        missing = sorted({n for c, n in test if c == cls} - set(names))
        if missing:
            sys.exit(f"{len(missing)} test images are missing from {src_dir}, for example {missing[0]}")
        for name in names:
            split = "test" if (cls, name) in test else "train"
            out_dir = os.path.join(args.dest, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(os.path.join(src_dir, name), os.path.join(out_dir, name))
            counts[(split, cls)] = counts.get((split, cls), 0) + 1

    ok = True
    for (split, cls), n in sorted(counts.items()):
        expected = EXPECTED_PER_CLASS[split]
        note = "" if n == expected else f"  (expected {expected})"
        ok = ok and n == expected
        print(f"{split}/{cls}: {n} images{note}")
    if not ok:
        sys.exit("Image counts differ from the split used in the project.")


if __name__ == "__main__":
    main()
