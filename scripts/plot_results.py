"""Regenerate plots from saved labels and the original DC data."""
import argparse
import json
from pathlib import Path

from vade.plotting import plot_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--data-dir", default="../data/DC")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=5000)
    args = p.parse_args()
    output = args.output_dir or Path("results") / args.dataset
    metadata = plot_dataset(args.dataset, args.data_dir, output, args.seed, args.sample_size)
    (output / "plot_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
