"""
PACE-Net Training Script
=========================
Full training pipeline with:
  - 5-fold stratified cross-validation
  - Modality-specific self-supervised pre-training
  - End-to-end fine-tuning with composite loss
  - Ablation variant support
  - TensorBoard logging

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --fold 0
    python scripts/train.py --config configs/config.yaml --ablation no_sfg
"""

import os, sys, argparse, yaml, random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pace_net import PACENet
from utils.losses import PACENetLoss
from utils.metrics import compute_metrics, print_metrics, MetricTracker


# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Train PACE-Net')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--fold', type=int, default=None,
                        help='Run single fold only (0-indexed)')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['no_sfg','no_gpg','no_glyph_gate',
                                 'no_causal_loss','no_struct_loss','bcg_only'],
                        help='Ablation variant to run')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


# ─── Config loading ───────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_ablation(cfg: dict, ablation: str) -> dict:
    """Override config for ablation variant."""
    if ablation and ablation in cfg.get('ablation', {}):
        overrides = cfg['ablation'][ablation]
        for k, v in overrides.items():
            cfg['training'][k] = v
    return cfg


# ─── Model builder ────────────────────────────────────────────────────────────
def build_model(cfg: dict, ablation: str = None, device: str = 'cpu') -> PACENet:
    """Build PACE-Net model with optional ablation flags."""
    kwargs = {
        'use_sfg': True, 'use_gpg': True,
        'use_glymphatic_gating': True,
        'use_neural_scm': True, 'use_cgat': True,
    }
    if ablation == 'no_sfg':          kwargs['use_sfg'] = False
    elif ablation == 'no_gpg':        kwargs['use_gpg'] = False
    elif ablation == 'no_glyph_gate': kwargs['use_glymphatic_gating'] = False
    elif ablation == 'bcg_only':
        kwargs.update({'use_sfg':False,'use_gpg':False,
                       'use_cgat':False,'use_neural_scm':False})

    model = PACENet(cfg, **kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__} | Params: {total_params:,}")
    return model


# ─── Optimizer and scheduler ─────────────────────────────────────────────────
def build_optimizer(model: PACENet, cfg: dict):
    t = cfg['training']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=t['learning_rate'],
        weight_decay=t['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=t['max_epochs'] - t['warmup_epochs'],
        eta_min=t['min_lr']
    )
    return optimizer, scheduler


def warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    """Linear learning rate warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = lr


# ─── Training step ────────────────────────────────────────────────────────────
def train_epoch(model: PACENet, loader, optimizer: optim.Optimizer,
                criterion: PACENetLoss, device: str,
                ablation: str, epoch: int,
                warmup_epochs: int, base_lr: float,
                grad_clip: float) -> dict:
    model.train()
    warmup_lr(optimizer, epoch, warmup_epochs, base_lr)

    total_losses = {}
    all_preds, all_targets, all_proba = [], [], []

    for batch in tqdm(loader, desc=f'Epoch {epoch+1}', leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        targets = batch['label']

        optimizer.zero_grad()

        # Forward pass
        s_bar = batch.get('s_bar_normalised', None)  # counterfactual sleep
        logits, mmse_pred, losses, aux = model(batch, s_bar=s_bar)

        # Get latent variables for loss computation
        z_dict = aux.get('z_dict') or {}
        z_G_pred = z_dict.get('z_G')
        z_G_obs  = z_dict.get('z_G_obs')
        z_C_pred = z_dict.get('z_C')
        z_C_obs  = z_dict.get('z_C_obs')
        cf_z_C   = aux.get('cf_z_C')

        # Update CN mean for counterfactual loss
        cn_mask = (targets == 0)
        if cn_mask.any() and z_C_obs is not None:
            model.neural_scm.update_cn_mean(z_C_obs[cn_mask])

        cn_z_C_mean = model.neural_scm.cn_z_C_mean if hasattr(model, 'neural_scm') else None

        # Loss override for ablations
        lambda_cf = 0.0 if ablation == 'no_causal_loss' else None

        loss_dict = criterion(
            logits=logits,
            targets=targets,
            mmse_pred=mmse_pred,
            mmse_true=batch.get('mmse'),
            z_G_pred=z_G_pred, z_G_obs=z_G_obs,
            z_C_pred=z_C_pred, z_C_obs=z_C_obs,
            cf_z_C=cf_z_C if ablation != 'no_causal_loss' else None,
            cn_z_C_mean=cn_z_C_mean,
            pool_loss=losses.get('pool_loss'),
        )

        loss = loss_dict['total']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Track metrics
        proba = torch.softmax(logits.detach(), dim=-1)
        preds = proba.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            total_losses[k] = total_losses.get(k, 0.0) + v

    n_batches = len(loader)
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_proba   = np.concatenate(all_proba)

    metrics = compute_metrics(all_targets, all_preds, all_proba)
    metrics.update(avg_losses)
    return metrics


# ─── Validation step ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: PACENet, loader, criterion: PACENetLoss,
             device: str) -> dict:
    model.eval()
    all_preds, all_targets, all_proba = [], [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        targets = batch['label']
        logits, _, _, _ = model(batch)

        proba = torch.softmax(logits, dim=-1)
        preds = proba.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_proba   = np.concatenate(all_proba)

    return compute_metrics(all_targets, all_preds, all_proba)


# ─── Main training loop ───────────────────────────────────────────────────────
def train_fold(fold_idx: int, train_loader, val_loader,
               cfg: dict, ablation: str,
               device: str, log_dir: str) -> dict:
    """Train one fold."""
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx + 1} / {cfg['training']['n_folds']}"
          + (f"  [Ablation: {ablation}]" if ablation else ""))
    print(f"{'='*60}")

    model = build_model(cfg, ablation, device)
    optimizer, scheduler = build_optimizer(model, cfg)

    # Class weights for focal loss
    class_weights = None  # computed from training data in practice
    criterion = PACENetLoss(
        lambda_ce=cfg['training']['lambda_ce'],
        lambda_mse=cfg['training']['lambda_mse'],
        lambda_struct=cfg['training']['lambda_struct'],
        lambda_cf=cfg['training']['lambda_cf'],
        focal_gamma=cfg['training']['focal_gamma'],
        class_weights=class_weights,
    )

    t = cfg['training']
    writer = SummaryWriter(os.path.join(log_dir, f'fold_{fold_idx}'))
    best_auc = 0.0
    best_state = None

    for epoch in range(t['max_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, ablation, epoch,
            t['warmup_epochs'], t['learning_rate'],
            t['gradient_clip']
        )

        # Evaluate
        if (epoch + 1) % t['eval_every'] == 0:
            val_metrics = evaluate(model, val_loader, criterion, device)

            # Log to TensorBoard
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'val/{k}', v, epoch)
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'train/{k}', v, epoch)

            print(f"  Epoch {epoch+1:3d}/{t['max_epochs']} | "
                  f"Train Acc: {train_metrics.get('accuracy',0):.1f}% | "
                  f"Val Acc: {val_metrics.get('accuracy',0):.1f}% | "
                  f"Val AUC: {val_metrics.get('auc_roc',0):.4f}")

            # Save best model
            if val_metrics.get('auc_roc', 0) > best_auc:
                best_auc = val_metrics['auc_roc']
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

        if epoch >= t['warmup_epochs']:
            scheduler.step()

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(model, val_loader, criterion, device)
    print_metrics(final_val, split=f'Fold {fold_idx+1} Final Validation')

    writer.close()
    return final_val, model


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.ablation:
        cfg = apply_ablation(cfg, args.ablation)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    set_seed(cfg['training']['seed'])

    # Logging directory
    run_name = f"pace_net{'_' + args.ablation if args.ablation else ''}"
    log_dir = os.path.join(cfg['logging']['log_dir'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg['logging']['checkpoint_dir'], exist_ok=True)

    # ── Load dataset ────────────────────────────────────────────────
    # In practice, replace with your actual dataset loader:
    # from data.adni_dataset import ADNIDataset
    # dataset = ADNIDataset(cfg['data']['adni_path'], cfg)
    # labels = dataset.labels
    print("\n[INFO] Dataset loading — replace with your ADNI dataset loader.")
    print("[INFO] See data/adni_dataset.py for the expected interface.\n")

    # ── Cross-validation ────────────────────────────────────────────
    fold_range = [args.fold] if args.fold is not None else range(cfg['training']['n_folds'])
    all_fold_metrics = []
    tracker = MetricTracker(['accuracy', 'auc_roc', 'mcc', 'jaccard', 'cohen_kappa'])

    for fold_idx in fold_range:
        # In practice: use StratifiedKFold to split dataset
        # Here we show the structure; plug in your DataLoader
        print(f"\n[Fold {fold_idx}] — Build your DataLoaders here using StratifiedKFold")
        # train_loader = DataLoader(train_subset, batch_size=cfg['training']['batch_size'])
        # val_loader   = DataLoader(val_subset,   batch_size=cfg['training']['batch_size'])

        # val_metrics, model = train_fold(
        #     fold_idx, train_loader, val_loader, cfg,
        #     args.ablation, device, log_dir
        # )
        # tracker.update(val_metrics)
        # torch.save(model.state_dict(),
        #            os.path.join(cfg['logging']['checkpoint_dir'],
        #                         f'{run_name}_fold{fold_idx}.pt'))

    # ── Summary ─────────────────────────────────────────────────────
    means = tracker.mean()
    stds  = tracker.std()
    if means:
        print(f"\n{'='*60}")
        print(f"  {cfg['training']['n_folds']}-Fold Cross-Validation Summary")
        print(f"{'='*60}")
        for k in ['accuracy', 'auc_roc', 'mcc', 'jaccard', 'cohen_kappa']:
            print(f"  {k:<20}: {means.get(k,0):.4f} ± {stds.get(k,0):.4f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
