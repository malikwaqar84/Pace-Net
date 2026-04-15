"""
PACE-Net Evaluation Script
===========================
Loads a trained checkpoint and evaluates on test data.
Produces all metrics reported in the paper.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
                                --config configs/config.yaml
"""

import os, sys, argparse, yaml
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pace_net import PACENet
from utils.metrics import compute_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--data_path', type=str, default='data/processed/')
    parser.add_argument('--dataset', type=str, default='adni',
                        choices=['adni', 'oasis', 'ukbb'])
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str, cfg: dict,
                         test_loader, device: str) -> dict:
    model = PACENet(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_targets, all_proba = [], [], []
    for batch in test_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits, _, _, _ = model(batch)
        proba = torch.softmax(logits, dim=-1)
        preds = proba.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch['label'].cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_proba   = np.concatenate(all_proba)

    return compute_metrics(all_targets, all_preds, all_proba)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device != 'auto':
        device = args.device

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[INFO] Evaluating checkpoint: {args.checkpoint}")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Device: {device}")
    print("\n[INFO] Load your test DataLoader here:")
    print("  from data.adni_dataset import ADNIDataset")
    print("  test_loader = DataLoader(ADNIDataset(args.data_path, cfg, split='test'))")
    print("\n  Then call:")
    print("  metrics = evaluate_checkpoint(args.checkpoint, cfg, test_loader, device)")
    print("  print_metrics(metrics)")


if __name__ == '__main__':
    main()
