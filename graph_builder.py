# ============================================================
# PACE-Net Configuration File
# ============================================================

# ----- Dataset -----------------------------------------------
data:
  adni_path: "data/processed/adni3/"
  oasis_path: "data/processed/oasis3/"
  ukbb_path: "data/processed/ukbiobank/"
  n_rois: 200                  # Schaefer atlas ROIs
  n_pvs_nodes: 24              # PVS regions (GPG nodes)
  n_sleep_channel_nodes: 8     # EEG channel nodes (SFG)
  n_sleep_biomarker_nodes: 8   # Sleep biomarker nodes (SFG)
  n_sleep_stages: 4            # N1, N2, N3, REM
  fmri_window_size: 60         # seconds, sliding window for dynamic FC
  fmri_window_step: 30         # seconds, step size
  eeg_bands: ["delta", "theta", "alpha", "sigma", "beta"]
  sigma_coherence_freq: [12, 15]  # Hz, spindle/sigma band
  classes: ["CN", "MCI", "AD"]
  n_classes: 3

# ----- Model Architecture ------------------------------------
model:
  # SFG Transformer
  sfg:
    n_layers: 4
    n_heads: 8
    hidden_dim: 128
    dropout: 0.3
    node_feat_dim: 32

  # GPG Transformer
  gpg:
    n_layers: 3
    n_heads: 4
    hidden_dim: 64
    dropout: 0.3
    node_feat_dim: 12

  # BCG Transformer
  bcg:
    n_layers: 6
    n_heads: 8
    hidden_dim: 256
    dropout: 0.3
    node_feat_dim: 20
    structural_threshold: 0.05   # DTI tractography threshold

  # Cross-Graph Attention Transformer (CGAT)
  cgat:
    n_heads: 8
    hidden_dim: 256
    dropout: 0.3
    fusion_token_dim: 256

  # DiffPool
  diffpool:
    n_clusters_1: 7             # Yeo-7 functional networks
    n_clusters_2: 1             # Final whole-brain embedding

  # Neural Structural Causal Model
  neural_scm:
    latent_dim: 128
    hidden_dims: [256, 128]
    dropout: 0.3

  # Classifier
  classifier:
    hidden_dims: [256, 128, 64]
    dropout: 0.3

# ----- Training -----------------------------------------------
training:
  n_folds: 5
  seed: 42
  max_epochs: 200
  pretrain_epochs: 50           # modality-specific self-supervised pretraining
  batch_size: 16
  num_workers: 4

  # Optimizer
  optimizer: "adamw"
  learning_rate: 1.0e-4
  weight_decay: 0.01
  gradient_clip: 1.0

  # Scheduler
  scheduler: "cosine_annealing"
  warmup_epochs: 10
  min_lr: 1.0e-6

  # Loss weights
  lambda_ce: 1.0               # cross-entropy classification loss
  lambda_mse: 0.4              # MMSE regression auxiliary loss
  lambda_struct: 0.6           # biological structural consistency loss
  lambda_cf: 0.2               # counterfactual regularisation loss
  focal_gamma: 2.0             # focal loss gamma
  class_weights: "inverse_freq" # handle class imbalance

# ----- Ablation Variants -------------------------------------
ablation:
  no_sfg:
    use_sfg: false
  no_gpg:
    use_gpg: false
  no_glyph_gate:
    use_glymphatic_gating: false
  no_causal_loss:
    lambda_cf: 0.0
  no_struct_loss:
    lambda_struct: 0.0
  bcg_only:
    use_sfg: false
    use_gpg: false
    use_cgat: false
    use_neural_scm: false

# ----- XAI --------------------------------------------------
xai:
  shap_n_samples: 100
  ig_n_steps: 50
  gradcam_layer: "bcg_transformer"
  counterfactual_n_steps: 1000
  counterfactual_lr: 0.01

# ----- Logging -----------------------------------------------
logging:
  log_dir: "logs/"
  checkpoint_dir: "checkpoints/"
  tensorboard: true
  save_every: 10               # save checkpoint every N epochs
  eval_every: 5
