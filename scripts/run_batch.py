"""Run the requested DC datasets in parallel, assigning jobs to devices."""
import argparse
import json
import subprocess
import sys
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vade.datasets import DATASETS, resolve_dataset


def run_one(name, args, device_pool):
    device = device_pool.get()
    try:
        return run_on_device(name, args, device)
    finally:
        device_pool.put(device)


def run_on_device(name, args, device):
    output = Path(args.results) / name
    command = [sys.executable, "-m", "vade.experiment", name,
               "--data-dir", args.data_dir, "--output-dir", str(output),
               "--device", device, "--seed", str(args.seed),
               "--threads", str(args.threads), "--batch-size", str(args.batch_size),
               "--epochs", str(args.epochs), "--pretrain-epochs", str(args.pretrain_epochs)]
    if args.overwrite:
        command.append("--overwrite")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    completed = status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed"
    if completed and not args.overwrite:
        return dict(dataset=name, device=device, status="skipped_completed", output=str(output))
    log = (output / "run.log").open("w")
    with log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    status = "completed" if process.returncode == 0 else "failed"
    return dict(dataset=name, device=device, status=status, returncode=process.returncode,
                output=str(output), log=str(output / "run.log"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="../data/DC")
    p.add_argument("--results", default="results")
    p.add_argument("--devices", nargs="+", default=["cuda:0"])
    p.add_argument("--workers", type=int, default=0, help="Defaults to one worker per device")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--datasets", nargs="*", choices=DATASETS, default=DATASETS)
    args = p.parse_args()
    devices = args.devices
    workers = args.workers or len(devices)
    if workers < 1 or workers > len(devices):
        raise ValueError("workers must be between 1 and the number of devices")
    missing = []
    for name in args.datasets:
        try:
            resolve_dataset(name, args.data_dir)
        except FileNotFoundError as error:
            missing.append(dict(dataset=name, status="missing", error=str(error)))
    available = [name for name in args.datasets if not any(item["dataset"] == name for item in missing)]
    Path(args.results).mkdir(parents=True, exist_ok=True)
    summary = missing[:]
    for item in missing:
        output = Path(args.results) / item["dataset"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "status.json").write_text(json.dumps(item, indent=2) + "\n")
    device_pool = queue.Queue()
    for device in devices[:workers]:
        device_pool.put(device)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run_one, name, args, device_pool): name
                for i, name in enumerate(available)}
        for future in as_completed(jobs):
            result = future.result()
            summary.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    # Preserve requested order in the manifest rather than completion order.
    order = {name: i for i, name in enumerate(args.datasets)}
    summary.sort(key=lambda item: order.get(item["dataset"], len(order)))
    (Path(args.results) / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    import csv
    with (Path(args.results) / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "status", "nmi", "ari", "f1_macro", "time_seconds"])
        writer.writeheader()
        for item in summary:
            metrics_path = Path(args.results) / item["dataset"] / "metrics.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            writer.writerow(dict(dataset=item["dataset"], status=item["status"], **metrics))
    if any(item.get("status") == "failed" for item in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
