"""A single reproducible DC clustering experiment."""
import argparse
import csv
import json
import os
import random
import sys
import contextlib
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .datasets import load_dataset
from .metrics import evaluate
from .model import VaDE

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def encode_all(model, x, batch_size):
    model.eval()
    return np.concatenate([model.encode(batch)[0].cpu().numpy() for batch in x.split(batch_size)])


@torch.no_grad()
def predict(model, x, batch_size):
    model.eval()
    return np.concatenate([model.log_responsibilities(model.encode(batch)[0]).argmax(-1).cpu().numpy()
                           for batch in x.split(batch_size)])


def initialize_mixture(model, x, seed, threads):
    latent = encode_all(model, x, 1024).astype(np.float64)
    k = model.config["n_clusters"]
    with threadpool_limits(limits=threads):
        means = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(latent).cluster_centers_
        variance = np.tile(latent.var(0, ddof=1) + 1e-3, (k, 1))
        mixture = GaussianMixture(n_components=k, covariance_type="diag", n_init=1,
                                  random_state=seed, reg_covar=1e-3, max_iter=100,
                                  means_init=means, weights_init=np.ones(k) / k,
                                  precisions_init=1 / variance).fit(latent)
    with torch.no_grad():
        model.cluster_means.copy_(torch.as_tensor(mixture.means_, dtype=x.dtype, device=x.device))
        model.cluster_log_vars.copy_(torch.as_tensor(mixture.covariances_, dtype=x.dtype, device=x.device).log())
        model.mixture_logits.zero_()
    print(json.dumps(dict(stage="gmm", converged=mixture.converged_, iterations=mixture.n_iter_,
                          lower_bound=mixture.lower_bound_)), flush=True)


def run(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed results already exist in {output}; use --overwrite to rerun")
    if args.overwrite:
        for name in ("metrics.json", "metrics.csv", "error.txt"):
            (output / name).unlink(missing_ok=True)
    seed_everything(args.seed, args.threads)
    device = torch.device(args.device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  plotting="original features, IDEC-style scatter/t-SNE", mixture_init="historical global covariance",
                  time_scope="data loading through final predicted labels; excludes metrics and output I/O")
    write_json(output / "config.json", config)
    write_json(output / "status.json", dict(status="running", pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S")))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    began = time.perf_counter()
    features, truth, encoded, scaler, metadata = load_dataset(args.dataset, args.data_dir)
    x = torch.from_numpy(features).to(device)
    model = VaDE(input_dim=x.shape[1], n_clusters=metadata["n_clusters"], latent_dim=args.latent_dim,
                 hidden_dims=args.hidden_dims, reconstruction="bce", alpha=1.).to(device)
    print(json.dumps(dict(stage="data", device=str(device), **metadata), ensure_ascii=False), flush=True)
    history = []
    mixture_ids = {id(p) for p in model.mixture_parameters()}
    network = [p for name, p in model.named_parameters() if id(p) not in mixture_ids]
    ae_parameters = [p for name, p in model.named_parameters()
                     if id(p) not in mixture_ids and not name.startswith("z_log_var")]
    ae_optimizer = torch.optim.Adam(ae_parameters, lr=args.pretrain_lr)
    for epoch in range(args.pretrain_epochs):
        model.train()
        total = 0.
        for ids in torch.randperm(len(x), device=device).split(args.batch_size):
            batch = x[ids]
            loss = model.reconstruction_loss(model.decoder(model.encode(batch)[0]), batch).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite AE loss")
            ae_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_parameters, 100., error_if_nonfinite=True)
            ae_optimizer.step()
            total += loss.item() * len(batch)
        record = dict(stage="pretrain", epoch=epoch + 1, loss=total / len(x))
        history.append(record)
        print(json.dumps(record), flush=True)
    with torch.no_grad():
        model.z_log_var.weight.zero_()
        model.z_log_var.bias.fill_(-4.)
    del ae_optimizer
    initialize_mixture(model, x, args.seed, args.threads)
    optimizer = torch.optim.Adam([dict(params=network, lr=args.lr),
                                  dict(params=model.mixture_parameters(), lr=args.gmm_lr)], eps=1e-4)
    for epoch in range(args.epochs):
        if epoch and epoch % 10 == 0:
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * .9, .0002)
        model.train()
        totals = dict(loss=0., reconstruction=0., kl=0.)
        for ids in torch.randperm(len(x), device=device).split(args.batch_size):
            batch = x[ids]
            loss, components = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite VaDE loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100., error_if_nonfinite=True)
            optimizer.step()
            for key, value in dict(loss=loss.detach(), **components).items():
                totals[key] += value.item() * len(batch)
        record = dict(stage="vade", epoch=epoch + 1, **{k: v / len(x) for k, v in totals.items()},
                      lr=[g["lr"] for g in optimizer.param_groups])
        history.append(record)
        print(json.dumps(record), flush=True)
    predictions = predict(model, x, 1024)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - began
    # Labels are used only now: evaluation after the timed unsupervised pipeline.
    scores, aligned, mapping = evaluate(truth, predictions)
    rounded = {k: f"{v:.4f}" for k, v in dict(**scores, time_seconds=elapsed).items()}
    write_json(output / "metrics.json", rounded)
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rounded))
        writer.writeheader()
        writer.writerow(rounded)
    np.save(output / "labels.npy", predictions)
    np.save(output / "labels_aligned.npy", aligned)
    np.save(output / "ground_truth.npy", truth)
    np.savetxt(output / "labels.csv", np.column_stack([np.arange(len(x)), predictions, aligned, encoded]),
               fmt="%d", delimiter=",", header="sample_index,cluster_id,aligned_class_id,true_class_id", comments="")
    write_json(output / "alignment.json", dict(cluster_to_encoded_class=mapping, class_values=metadata["class_values"]))
    write_json(output / "dataset.json", metadata)
    np.savez(output / "normalization.npz", data_min=scaler.data_min_, data_max=scaler.data_max_,
             scale=scaler.scale_, offset=scaler.min_)
    (output / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    torch.save(dict(model_config=model.config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=args.epochs, config=config), output / "model.pt")
    from .plotting import plot_dataset
    plot_metadata = plot_dataset(args.dataset, args.data_dir, output, args.seed, args.plot_samples, args.threads)
    write_json(output / "plot_config.json", plot_metadata)
    write_json(output / "status.json", dict(status="completed", finished=time.strftime("%Y-%m-%d %H:%M:%S")))
    print(json.dumps(dict(stage="completed", **rounded)), flush=True)


def parser():
    p = argparse.ArgumentParser(description="Train VaDE on a DC dataset from scratch")
    p.add_argument("dataset")
    p.add_argument("--data-dir", type=Path, default=ROOT.parent / "data/DC")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--pretrain-lr", type=float, default=.001)
    p.add_argument("--lr", type=float, default=.002)
    p.add_argument("--gmm-lr", type=float, default=.002)
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[500, 500, 2000])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--plot-samples", type=int, default=5000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_dir = args.output_dir or ROOT / "results" / args.dataset
    if min(args.threads, args.batch_size, args.epochs, args.pretrain_epochs, args.latent_dim, *args.hidden_dims) <= 0:
        raise ValueError("Dimensions, batch size, threads and epoch counts must be positive")
    if min(args.lr, args.gmm_lr, args.pretrain_lr) <= 0:
        raise ValueError("Learning rates must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # The batch launcher captures stdout; standalone runs also retain a log.
        class Tee:
            def __init__(self, terminal, log):
                self.terminal, self.log = terminal, log

            def write(self, text):
                self.terminal.write(text)
                self.log.write(text)

            def flush(self):
                self.terminal.flush()
                self.log.flush()

        with (args.output_dir / "train.log").open("w") as log:
            tee = Tee(sys.stdout, log)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                run(args)
    except Exception as error:
        if args.output_dir.exists():
            write_json(args.output_dir / "status.json", dict(status="failed", error=str(error)))
            (args.output_dir / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
