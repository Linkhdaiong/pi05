#!/usr/bin/env python3
"""Score PI0.5 checkpoints with the official LeRobot validation loader and flow loss.

This is an offline teacher-forced flow-matching *loss*, not action MAE or robot success.
No training, optimizer, or checkpoint writes occur. Test access must be explicitly requested.
"""

import argparse
import contextlib
import io
import json
import math
import os
import hashlib
import time
import traceback
from collections import defaultdict
from pathlib import Path

CAMERAS = {
    "observation.images.side": "observation.images.base_0_rgb",
    "observation.images.front": "observation.images.left_wrist_0_rgb",
}
DEFAULT_SPLITS = Path("/path/to/workdir/datasets/pi05_block_sidebox_v2_splits")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-root", type=Path, default=DEFAULT_SPLITS / "val")
    p.add_argument("--train-root", type=Path, default=DEFAULT_SPLITS / "train")
    p.add_argument("--checkpoint", type=Path, action="append", required=True,
                   help="Repeat for each checkpoint's pretrained_model dir (or pi05_base)")
    p.add_argument("--base-train-stats", action="store_true",
                   help="For pi05_base ONLY: explicitly normalize with train/meta/stats.json; "
                        "base ships no saved normalizer stats, so its loss is provisional")
    p.add_argument("--tokenizer-path", type=Path,
                   help="Authorized local PaliGemma tokenizer directory; overrides only the tokenizer source")
    p.add_argument("--max-per-episode", type=int, default=2)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--max-total", type=int, default=38)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-frames", action="store_true", help="Evaluate every frame, streaming without an image cache")
    p.add_argument("--stream", action="store_true", help="Stream sparse samples too (for bounded preflight)")
    p.add_argument("--output-dir", type=Path, help="Persist protocol, per-frame scores, status, and final result")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data-only", action="store_true", help="Decode selected val items; no weights/GPU")
    a = p.parse_args()
    if min(a.max_per_episode, a.stride, a.max_total) <= 0:
        p.error("sample caps and stride must all be positive")
    if not (a.val_root / "meta" / "info.json").is_file():
        p.error(f"not an existing local LeRobot validation dataset: {a.val_root}")
    # The train root is only required to supply stats for an untrained base.
    # Fine-tuned checkpoints carry their own preprocessing statistics.
    if a.tokenizer_path is not None and not any((a.tokenizer_path / n).is_file() for n in ("tokenizer.model", "tokenizer.json")):
        p.error(f"tokenizer directory needs tokenizer.model or tokenizer.json: {a.tokenizer_path}")
    if (a.all_frames or a.resume) and a.output_dir is None:
        p.error("--all-frames/--resume requires --output-dir")
    if a.output_dir and len(a.checkpoint) != 1:
        p.error("Durable output requires exactly one frozen checkpoint")
    for checkpoint in a.checkpoint:
        if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
            p.error(f"checkpoint needs config.json and model.safetensors: {checkpoint}")
    return a


def samples(meta, stride, cap_episode, cap_total, all_frames=False):
    """Select sparse samples or every frame, keeping complete episode boundaries."""
    if all_frames:
        selected = [(int(ep["episode_index"]), frame, int(ep["dataset_from_index"]) + frame)
                    for ep in sorted(meta.episodes, key=lambda x: x["episode_index"])
                    for frame in range(int(ep["length"]))]
        if len(selected) != meta.total_frames or [x[2] for x in selected] != list(range(meta.total_frames)):
            raise ValueError("Full-frame metadata must cover every global index exactly once")
        return selected
    out = []
    for episode in sorted(meta.episodes, key=lambda x: x["episode_index"]):
        length = int(episode["length"])
        # Spread samples across each episode's timeline, not just its idle prefix.
        frames = sorted({min(length - 1, round((j + 1) * length / (cap_episode + 1) / stride) * stride)
                         for j in range(cap_episode)})
        for frame in frames:
            if len(out) >= cap_total:
                break
            out.append((int(episode["episode_index"]), frame,
                        int(episode["dataset_from_index"]) + frame))
        if len(out) >= cap_total:
            break
    if not out:
        raise ValueError("No validation samples found")
    return out


class ScoreJournal:
    """Append completed frame scores durably; resume only an identical manifest."""

    def __init__(self, root, manifest, resume):
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        path = root / "samples.jsonl"
        if manifest_path.exists():
            if not resume:
                raise ValueError("Output already exists; use --resume for the identical evaluation")
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Resume manifest differs from current model/data/protocol")
        else:
            if resume or path.exists():
                raise ValueError("Missing resume manifest or orphaned sample journal")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self.completed = {}
        if path.exists():
            with path.open("r+b") as f:
                while True:
                    start = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        f.truncate(start)
                        break
                    row = json.loads(line)
                    idx = row["global_index"]
                    if idx in self.completed or not math.isfinite(row["flow_loss"]):
                        raise ValueError("Duplicate or nonfinite persisted score")
                    self.completed[idx] = row
        self.file = path.open("a", buffering=1)

    def append(self, row):
        idx = row["global_index"]
        if idx in self.completed or not math.isfinite(row["flow_loss"]):
            raise ValueError("Duplicate or nonfinite score")
        self.file.write(json.dumps(row, sort_keys=True) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.completed[idx] = row

    def close(self):
        self.file.close()


def check_schema(meta):
    features = meta.features
    if any(key not in features for key in (*CAMERAS, "observation.state", "action")):
        raise ValueError(f"val feature mismatch; expected {list(CAMERAS)} + state/action; got {list(features)}")
    # Physical labels were confirmed by the user: front is the left wrist view.
    if len(set(CAMERAS.values())) != len(CAMERAS):
        raise ValueError("camera rename collision")


def checkpoint_stats_mode(checkpoint, allow_base):
    """Do not infer training provenance from an arbitrary nearby dataset."""
    pre = checkpoint / "policy_preprocessor.json"
    if not pre.is_file():
        raise ValueError(f"No saved policy preprocessor: {pre}")
    steps = json.loads(pre.read_text())["steps"]
    idx = next((i for i, step in enumerate(steps)
                if step["registry_name"] == "normalizer_processor"), None)
    if idx is None:
        raise ValueError(f"No checkpoint normalizer: {pre}")
    normal = steps[idx]
    stats_file = checkpoint / f"policy_preprocessor_step_{idx}_normalizer_processor.safetensors"
    # The base distribution contains no normalizer buffers and an empty feature spec.
    is_base = not normal["config"].get("features") and not stats_file.is_file()
    if is_base:
        if not allow_base:
            raise ValueError(f"{checkpoint}: base lacks saved stats; pass --base-train-stats "
                             "to explicitly use train-only stats as a provisional base score")
        return "provisional_train_stats"
    if not stats_file.is_file():
        raise ValueError(f"{checkpoint}: saved normalizer feature config but missing {stats_file.name}")
    return "checkpoint_saved_stats"


def load_pipeline(cfg, checkpoint, mode, train_meta, tokenizer_path):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor.rename_processor import rename_stats

    if mode == "provisional_train_stats":
        if train_meta is None:
            raise ValueError("Base checkpoint requires train-only normalization statistics")
        # The official training code builds the base preprocessor from train-only stats.
        return make_pre_post_processors(cfg, dataset_stats=rename_stats(train_meta.stats, CAMERAS))[0]
    # Native fine-tuned checkpoint: retain its *saved* normalizer buffers, never val stats.
    pre_overrides = {
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {
            "features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": CAMERAS},
    }
    if tokenizer_path is not None:
        pre_overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_path.resolve())}
    if cfg.use_relative_actions:
        pre_overrides["relative_actions_processor"] = {
            "enabled": True, "exclude_joints": cfg.relative_exclude_joints,
            "action_names": cfg.action_feature_names,
        }
    return make_pre_post_processors(cfg, pretrained_path=checkpoint,
                                    preprocessor_overrides=pre_overrides)[0]


def run_eval(a):
    import torch
    import lerobot.policies.pi05.configuration_pi05  # noqa: F401; registers draccus type
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.collate import lerobot_collate_fn

    val_meta = LeRobotDatasetMetadata("local/pi05_val", root=a.val_root)
    check_schema(val_meta)
    selected = samples(val_meta, a.stride, a.max_per_episode, a.max_total, a.all_frames)
    print("SPLIT", a.val_root.resolve(), "frames", val_meta.total_frames,
          "episodes", val_meta.total_episodes, "selected_count", len(selected),
          "selected_edges", selected[:2] + selected[-2:], flush=True)
    print("CAMERA_MAP", json.dumps(CAMERAS, sort_keys=True),
          "missing_right_wrist=official_PI05_zero_mask", flush=True)
    if not a.data_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for checkpoint evaluation")
    results = []
    raw = None
    train_meta = None
    for cp in a.checkpoint:
        cfg = PreTrainedConfig.from_pretrained(cp)
        if cfg.type != "pi05":
            raise ValueError(f"Only PI05 checkpoints supported: {cp}")
        cfg.device = "cuda" if not a.data_only else "cpu"
        cfg.pretrained_path = cp.resolve()
        if cfg.use_visual_memory or cfg.use_proprioceptive_memory:
            raise ValueError("This bounded sampler does not support history/memory checkpoints")
        if len(cfg.action_delta_indices) != cfg.chunk_size:
            raise ValueError("Unexpected PI05 action window configuration")
        if not set(CAMERAS.values()).issubset(cfg.image_features):
            raise ValueError(f"Checkpoint expects different camera keys: {cfg.image_features}")
        delta = resolve_delta_timestamps(cfg, val_meta, CAMERAS)
        if "action" not in delta or len(delta["action"]) != cfg.chunk_size:
            raise ValueError("Missing full action chunk in official delta timestamp resolver")
        ds = LeRobotDataset("local/pi05_val", root=a.val_root, delta_timestamps=delta,
                            video_backend="pyav", return_uint8=True)
        # Sparse comparisons may cache decoded items; full evaluation streams them.
        streaming = a.all_frames or a.stream
        journal = None
        if a.output_dir is not None:
            paths = sorted(set([*cp.rglob("*"), *a.val_root.rglob("*")]))
            identity = [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns]
                        for p in paths if p.is_file()]
            manifest = {"checkpoint": str(cp.resolve()), "split": str(a.val_root.resolve()),
                        "selected": selected, "seed": a.seed, "stats_mode": "checkpoint_saved_stats",
                        "camera_map": CAMERAS, "batch_size": 1, "amp": "bf16",
                        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "files": identity, "tokenizer_override": str(a.tokenizer_path.resolve()) if a.tokenizer_path else None}
            # JSON round-trip normalizes tuples before comparison on resume.
            journal = ScoreJournal(a.output_dir, json.loads(json.dumps(manifest)), a.resume)
            for idx, row in journal.completed.items():
                if idx >= len(ds) or (row["episode"], row["frame"], idx) not in selected:
                    raise ValueError("Persisted frame is outside the selected protocol")
            print("RESUME_COMPLETED", len(journal.completed), flush=True)
        if raw is None and not streaming:
            raw = []
            for ep, frame, idx in selected:
                item = ds[idx]
                for k in CAMERAS:
                    if item[k].dtype != torch.uint8 or item[k].ndim != 3:
                        raise ValueError(f"Official loader gave unexpected image: {k}: {item[k].shape}")
                if item["action"].shape[0] != cfg.chunk_size:
                    raise ValueError(f"Action window shape mismatch: {item['action'].shape}")
                raw.append(item)
            print("DATA_DECODE_OK", len(raw), "first_image", tuple(raw[0][next(iter(CAMERAS))].shape),
                  "action", tuple(raw[0]["action"].shape), flush=True)
        if a.data_only:
            if streaming:
                for ep, frame, idx in selected:
                    item = ds[idx]
                    if item["action"].shape[0] != cfg.chunk_size:
                        raise ValueError("Invalid full action chunk")
                print("DATA_DECODE_OK", len(selected), flush=True)
            if journal:
                journal.close()
            continue
        mode = checkpoint_stats_mode(cp, a.base_train_stats)
        if mode == "provisional_train_stats" and train_meta is None:
            if a.train_root.resolve() == a.val_root.resolve():
                raise ValueError("Base normalization must use train, not val")
            if not (a.train_root / "meta" / "info.json").is_file():
                raise ValueError(f"Base checkpoint requires local train dataset: {a.train_root}")
            train_meta = LeRobotDatasetMetadata("local/pi05_train", root=a.train_root)
            check_schema(train_meta)
            print("NORMALIZATION_TRAIN_SOURCE", a.train_root.resolve(),
                  "frames", train_meta.total_frames, flush=True)
        # Check the actual tokenizer source before allocating the multi-GB model.
        from transformers import AutoTokenizer
        if a.tokenizer_path is not None:
            tokenizer_name = str(a.tokenizer_path.resolve())
        else:
            name = next((s["config"].get("tokenizer_name")
                         for s in json.loads((cp / "policy_preprocessor.json").read_text())["steps"]
                         if s["registry_name"] == "tokenizer_processor"), cfg.text_tokenizer_name)
            # Fine-tuned checkpoints may bundle it as a relative "tokenizer" dir.
            tokenizer_name = str((cp / name).resolve()) if (cp / name).is_dir() else name
        try:
            AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=Path(tokenizer_name).is_dir())
        except OSError as exc:
            raise RuntimeError(f"Cannot load required tokenizer {tokenizer_name}") from exc
        print("TOKENIZER_ACCESS_OK", tokenizer_name, flush=True)
        if mode == "provisional_train_stats" and a.tokenizer_path is not None:
            # Base uses train-only stats and a locally provided tokenizer, in memory.
            cfg.text_tokenizer_name = tokenizer_name
        # Never silently score random weights: the OpenPI base loader catches its own exceptions.
        from lerobot.policies.factory import make_policy
        # LeRobot 0.6.2 (297774f): same batch conversion/rename as trainer.
        from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            policy = make_policy(cfg=cfg, ds_meta=val_meta, rename_map=CAMERAS)
        load_message = stream.getvalue()
        if ("All keys loaded successfully!" not in load_message
                and "✓ Loaded state dict from model.safetensors" not in load_message
                and not policy.use_native_pretrained_loader):
            raise RuntimeError(f"PI05 base failed to load all weights: {load_message[-2000:]}")
        if "Warning: Could not load" in load_message or "Returning model without" in load_message:
            raise RuntimeError(f"Checkpoint silently failed weight load: {load_message[-2000:]}")
        print("WEIGHTS_LOAD", cp.resolve(), load_message[-450:].strip(), flush=True)
        policy.eval()
        preprocessor = load_pipeline(cfg, cp, mode, train_meta, a.tokenizer_path)
        totals = defaultdict(list)
        if journal:
            for row in journal.completed.values():
                totals[row["episode"]].append(row["flow_loss"])
        score_started = time.monotonic()
        initial_completed = sum(map(len, totals.values()))
        print("CHECKPOINT", cp.resolve(), "stats_mode", mode, "seed", a.seed,
              "amp=bf16", "batch=1", flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for sample_number, (ep, frame, idx) in enumerate(selected):
                if journal and idx in journal.completed:
                    continue
                item = ds[idx] if streaming else raw[sample_number]
                if item["action"].shape[0] != cfg.chunk_size:
                    raise ValueError("Action chunk mismatch")
                # Reset both CPU and CUDA RNG before *each* forward to share flow noise/time.
                torch.manual_seed(a.seed + idx)
                batch = lerobot_collate_fn([item])
                batch = _preprocess_dataset_batch(batch, val_meta.camera_keys, CAMERAS, preprocessor)
                loss, _ = policy(batch)
                value = float(loss.item())
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite flow loss: checkpoint={cp}, index={idx}")
                totals[ep].append(value)
                if journal:
                    journal.append({"episode": ep, "frame": frame, "global_index": idx, "flow_loss": value})
                count = sum(map(len, totals.values()))
                if count == initial_completed + 1 or count % 50 == 0 or count == len(selected):
                    elapsed = time.monotonic() - score_started
                    print("PROGRESS", count, "/", len(selected), "elapsed_s", round(elapsed, 3),
                          "seconds_per_frame", round(elapsed / (count - initial_completed), 4),
                          "peak_gpu_bytes", torch.cuda.max_memory_allocated(), flush=True)
                    if a.output_dir:
                        (a.output_dir / "status.json").write_text(json.dumps({"status": "scoring", "completed": count,
                            "expected": len(selected), "updated_at": time.time(), "elapsed_s": elapsed}) + "\n")
                print("SAMPLE", cp.name, "episode", ep, "frame", frame,
                      "global_index", idx, "flow_loss", value, flush=True)
        sample_values = [x for vals in totals.values() for x in vals]
        report = {"checkpoint": str(cp.resolve()), "stats_mode": mode,
                  "samples": len(sample_values), "flow_loss_mean": sum(sample_values) / len(sample_values),
                  "episodes": {str(ep): {"samples": len(vals), "flow_loss_mean": sum(vals) / len(vals)}
                               for ep, vals in sorted(totals.items())}}
        if report["samples"] != len(selected):
            raise ValueError("Incomplete result coverage")
        if journal:
            journal.close()
        report.update({"split": str(a.val_root.resolve()), "all_frames": a.all_frames, "seed": a.seed,
                       "elapsed_scoring_s": time.monotonic() - score_started,
                       "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        if a.output_dir:
            (a.output_dir / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        results.append(report)
        print("RESULT", json.dumps(report, sort_keys=True), flush=True)
        del policy, preprocessor
        torch.cuda.empty_cache()
    if results:
        best = min(results, key=lambda r: r["flow_loss_mean"])
        print("LOWEST_FLOW_LOSS", best["checkpoint"], best["flow_loss_mean"], flush=True)
        if len({r["stats_mode"] for r in results}) != 1:
            print("CAUTION: stats provenance differs; base provisional score is not a "
                  "like-for-like checkpoint selection metric", flush=True)
    else:
        print("DATA_ONLY_OK; no checkpoint weights loaded", flush=True)


def main():
    a = parse_args()
    try:
        run_eval(a)
    except BaseException as exc:
        if a.output_dir and a.output_dir.exists():
            (a.output_dir / "status.json").write_text(json.dumps({"status": "failed",
                "error": repr(exc), "traceback": traceback.format_exc(), "updated_at": time.time()}) + "\n")
        raise
    else:
        if a.output_dir:
            (a.output_dir / "status.json").write_text(json.dumps({"status": "completed",
                "result": str(a.output_dir / "result.json"), "updated_at": time.time()}) + "\n")


if __name__ == "__main__":
    main()
