# Stage23 no-direction FP32 server run

This branch is based on the Stage23 experiment configuration from 2026-09-25. It keeps the 1024-to-256 direct resize, learning rates, topology/skeleton settings, and disables direction prediction, direction loss, and direction-conditioned/multi-hop decoder attention. Training runs in FP32 (`--amp_dtype none`).

```bash
git clone --branch codex/stage23-no-direction-fp32-20260930 https://github.com/ganjiajia77-maker/SwinNet.git SwinNet-stage23-no-direction-fp32
cd SwinNet-stage23-no-direction-fp32

REPO="$PWD"
DATA=/home/gjj/Swin-Unet-main/data1
MODEL_ROOT=/home/gjj/Swin-Unet-main
RUN=data1_58d60dd_stage23softske_pw2_pairlinear_con3_dir02_direct256_h2h3residual_focal1_fp32_nodir_100e_20260930
OUTDIR="$MODEL_ROOT/model_out/$RUN"
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=1 \
OPENCV_LOG_LEVEL=ERROR \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u train_image.py \
  --root_path "$DATA" \
  --output_dir "$MODEL_ROOT/model_out" \
  --run_name "$RUN" \
  --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
  --pretrain_ckpt "$MODEL_ROOT/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth" \
  --warm_start_ckpt "$MODEL_ROOT/model_out/data1_58d60dd_stage23softske_pw2_pairlinear_con3_dir02_direct256_60e_focal1_bf16_20260925_2/best.pth" \
  --pretrained_lr 5e-5 \
  --pretrained_min_lr 5e-6 \
  --new_lr 2e-4 \
  --new_min_lr 1e-5 \
  --warmup_epochs 10 \
  --amp_dtype none \
  --structure_profile stage23_boundary_0626 \
  --enable_highres_structure_stream \
  --enable_global_topology \
  --global_topology_max_nodes 32 \
  --global_topology_heads 4 \
  --global_topology_alpha_max 0.05 \
  --highres_structure_fuse_stages stage23 \
  --highres_structure_fusion_mode stage23 \
  --final_topology_eta_init 0.0 \
  --final_gap_rho_init 0.0 \
  --img_size 256 \
  --source_patch_size 1024 \
  --direct_resize_train \
  --overlap_stride 256 \
  --max_epochs 100 \
  --val_interval 5 \
  --batch_size 4 \
  --num_workers 4 \
  --stage2_skeleton_weight 0.008 \
  --stage3_skeleton_weight 0.012 \
  --stage2_skeleton_gradient_ratio 0.5 \
  --stage3_skeleton_gradient_ratio 0.5 \
  --stage3_gate_topology_gradient_ratio 0.0 \
  --skeleton_pos_weight 2.0 \
  --highres_structure_skeleton_weight 0.0 \
  --stage_connectivity_factor 3.0 \
  --stage_direction_factor 0.0 \
  --road_attention_weight 0.0 \
  --masked_connectivity_center_experiment \
  --connectivity_pos_weight 2.0 \
  --connectivity_focal_gamma 1.5 \
  --edge_contrastive_margin 0.1 \
  --directional_pos_weight_cardinal 1.0 \
  --directional_pos_weight_diagonal 2.5 \
  --surface_focal_gamma 1.0 \
  --no-use_ema \
  --threshold 0.2 \
  --seed 1234
```

After training, sweep thresholds on validation:

```bash
python -u threshold_sweep_val_fixed.py \
  --root_path "$DATA" \
  --model_path "$OUTDIR/best.pth" \
  --split val \
  --img_size 256 \
  --source_patch_size 1024 \
  --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
  --structure_profile stage23_boundary_0626 \
  --enable_highres_structure_stream \
  --enable_global_topology \
  --global_topology_max_nodes 32 \
  --global_topology_heads 4 \
  --global_topology_alpha_max 0.05
```

Run final test-set inference using the chosen threshold (replace `0.30` with the validation-selected value):

```bash
python -u test_image.py \
  --root_path "$DATA" \
  --model_path "$OUTDIR/best.pth" \
  --output_dir "$MODEL_ROOT/predictions/$RUN" \
  --split test \
  --img_size 256 \
  --source_patch_size 1024 \
  --threshold 0.30 \
  --cfg "$REPO/configs/swin_tiny_patch4_window7_224_lite.yaml" \
  --structure_profile stage23_boundary_0626 \
  --enable_highres_structure_stream \
  --enable_global_topology \
  --global_topology_max_nodes 32 \
  --global_topology_heads 4 \
  --global_topology_alpha_max 0.05
```
