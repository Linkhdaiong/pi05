#!/usr/bin/env python3
"""Run the agreed official PI05 training CLI, then score five EMA checkpoints.

No automatic restart, checkpoint deletion, test-set access, or shutdown.
"""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/path/to/workdir')
ENV = ROOT / 'envs/l'
OUTPUT = ROOT / 'outputs/pi05_sidebox_base_trainonly_v1'
LOGS = ROOT / 'logs/pi05_sidebox_base_trainonly_v1'
EVALUATOR = ROOT / 'projects/eval_pi05_checkpoints.py'
STEPS = list(range(2000, 10001, 2000))
CAMERAS = {
    'observation.images.side': 'observation.images.base_0_rgb',
    'observation.images.front': 'observation.images.left_wrist_0_rgb',
}
TRAIN = [
    str(ENV / 'bin/lerobot-train'),
    '--dataset.repo_id=local/pi05_sidebox_train',
    f'--dataset.root={ROOT}/datasets/pi05_block_sidebox_v2_splits/train',
    '--dataset.eval_split=0', '--dataset.use_imagenet_stats=false',
    '--dataset.video_backend=torchcodec', '--dataset.image_transforms.enable=true',
    f'--policy.path={ROOT}/models/pi05_base', '--policy.device=cuda',
    '--policy.dtype=bfloat16', '--policy.gradient_checkpointing=true',
    '--policy.push_to_hub=false', '--rename_map=' + json.dumps(CAMERAS),
    '--accelerator.mixed_precision=bf16', '--batch_size=32', '--num_workers=4',
    '--dataloader_multiprocessing_context=spawn', '--steps=10000',
    '--save_checkpoint=true', '--save_freq=2000', '--log_freq=100',
    '--eval_steps=0', '--env_eval_freq=0', '--ema.enable=true', '--ema.decay=0.99',
    '--wandb.enable=false', '--job_name=pi05_sidebox_base_trainonly_v1',
    f'--output_dir={OUTPUT}',
]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'status': 'starting', 'output_dir': str(OUTPUT),
            'log_dir': str(LOGS), 'expected_steps': STEPS,
            'evaluated': [], 'train_command': shlex.join(TRAIN),
        }
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PATH'] = str(ENV / 'bin') + os.pathsep + env.get('PATH', '')

        def status(value, **extra):
            state.update(status=value, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
            write_json(LOGS / 'status.json', state)
            print('PIPELINE_STATUS', value, flush=True)

        def run(command, log_path):
            with log_path.open('x') as log:
                print('COMMAND', shlex.join(command), flush=True)
                log.write('COMMAND ' + shlex.join(command) + '\n')
                log.flush()
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    code = proc.wait()
                if code:
                    raise RuntimeError(f'Command exited {code}; see {log_path}')

        stage = 'training'
        try:
            if OUTPUT.exists():
                raise FileExistsError(f'Refusing to overwrite an existing run: {OUTPUT}')
            status('training')
            run(TRAIN, LOGS / 'train.log')
            status('training_completed')
            stage = 'evaluating'
            results = []
            for step in STEPS:
                cp = OUTPUT / 'checkpoints' / f'{step:06d}' / 'pretrained_model_ema'
                for filename in ('model.safetensors', 'config.json',
                                 'policy_preprocessor.json', 'policy_postprocessor.json'):
                    if not (cp / filename).is_file():
                        raise FileNotFoundError(cp / filename)
                saved = json.loads((cp.parent / 'training_state/training_step.json').read_text())
                if saved['step'] != step:
                    raise ValueError(f'Checkpoint step mismatch: {cp}')
                status('evaluating', current_checkpoint=str(cp), current_step=step)
                log_path = LOGS / f'eval_ema_{step:06d}.log'
                command = [str(ENV / 'bin/python'), str(EVALUATOR),
                           '--checkpoint', str(cp),
                           '--val-root', str(ROOT / 'datasets/pi05_block_sidebox_v2_splits/val'),
                           '--train-root', str(ROOT / 'datasets/NO_SUCH_TRAIN'),
                           '--tokenizer-path', str(ROOT / 'models/paligemma-3b-pt-224-tokenizer'),
                           '--max-total', '38', '--max-per-episode', '2', '--seed', '42']
                run(command, log_path)
                records = [json.loads(line.removeprefix('RESULT '))
                           for line in log_path.read_text().splitlines() if line.startswith('RESULT ')]
                if len(records) != 1 or records[0]['samples'] != 38:
                    raise ValueError(f'Expected one 38-sample result: {log_path}')
                record = records[0]
                if record['stats_mode'] != 'checkpoint_saved_stats':
                    raise ValueError(f'Expected checkpoint-saved statistics: {log_path}')
                record['step'] = step
                results.append(record)
                state['evaluated'] = results
                write_json(LOGS / 'eval_results.json', results)
                status('checkpoint_evaluated')
            best = min(results, key=lambda row: row['flow_loss_mean'])
            status('completed', best_by_val_flow_loss=best)
            print('BEST_EMA_CHECKPOINT', best['checkpoint'], best['flow_loss_mean'], flush=True)
        except BaseException as exc:
            status('failed_' + stage, error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
