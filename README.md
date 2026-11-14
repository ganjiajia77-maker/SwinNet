# Road extraction with decoder Stage2 global topology

This release contains the encoder-H model evaluated at `ab60f17f200952b361e88650913dc89ce9270b8c`. The global topology module runs after decoder Stage2 refinement and before Stage3 skip fusion.

## Model reading order

1. [Main network](networks/swin_transformer_unet_skip_expand_decoder_sys.py) defines the Swin encoder, bottleneck and decoder. Start with `SwinTransformerSys.forward`, then follow `forward_features`, `forward_up_features` and `up_x4`.
2. [Encoder H exchange](networks/encoder_h_postmerge_phase.py) keeps H on a 128 × 128 grid with 64 channels for a 512 × 512 input. Before each of the first three patch merges, encoder features update H through a projected, resized residual. After each merge, pixel unshuffle groups H subpositions into channels, and a projected residual updates the merged encoder feature. The updated H stream continues into the decoder.
3. [Local structure refinement](networks/skeleton_guided_head.py) implements the decoder structure blocks. The Stage2 block produces the updated main feature, H1 and the local connectivity logits C2. The shared surface head reads the Stage2 main feature to produce a detached road prior P2.
4. [Global topology](networks/keypoint_global_topology.py) scores candidate anchors with normalized H1 feature magnitude and P2. The default topology configuration uses up to 256 candidates and 32 sampled anchors. It uses relative geometry and sampled local connectivity in node attention, then lets the main feature grid attend to the refined nodes. A P2-gated, learnably scaled residual updates the Stage2 main feature. Stage3 receives that updated feature at its normal skip fusion. C2 is detached on the global module's input path. Final upsampling and the G3 surface fusion produce the road output.
5. [Training losses](losses/road_losses.py) contains the road and structure objectives. [Training configuration](h0_phase_training.py) resolves the settings from the original full H0 checkpoint's saved arguments. Fresh training loads the SwinV2/ImageNet initialization.

The global module exchanges context over the current inference tile. The [anchor diagnostic](scripts/diagnose_h0_global_anchor_gaps.py) measures whether sampled anchors cover both sides of fixed gaps. Anchor coverage, gap closure and graph-level connectivity are separate measurements.

## Reproduction

Install the dependencies in `requirements.txt` and supply the Massachusetts dataset, the original full H0 checkpoint used for configuration, and the corresponding SwinV2 pretrained file. Use checkpoints from a trusted source.

```bash
export DATA=/path/to/Massachusetts/tiff
export RESULTS=/path/to/run_outputs
export PRETRAIN=/path/to/swinv2_tiny_patch4_window8_256.pth
export H0_REFERENCE=/path/to/original_h0/best.pth
export PHASE_RUN_NAME=stage2_global
bash scripts/run_h0_encoder_phase.sh check
bash scripts/run_h0_encoder_phase.sh smoke
bash scripts/run_h0_encoder_phase.sh train
```

The measured setup uses 80 epochs, batch size 2, FP32, no EMA and one random 512 × 512 crop per training image per epoch. The data workers restart every epoch. The launcher checks crop sampling and initial model state before starting training.

```bash
python scripts/evaluate_h0_encoder_phase.py \
  --checkpoint "$RESULTS/$PHASE_RUN_NAME/best.pth" \
  --root_path "$DATA" --output_dir "$RESULTS/${PHASE_RUN_NAME}_eval" \
  --command all --topo_tool /path/to/compare_predictions.py
```

Evaluation selects the IoU threshold on 14 validation images and applies it to 49 test images. It uses native 1500 × 1500 images, 512 × 512 windows at stride 256 and triangular logit blending. `--topo_tool` expects the unified six-metric evaluator from revision `4b7d37a`. Use `--command infer` for segmentation metrics without that external evaluator.

The released evaluator loads the original Stage2 checkpoint by discarding its single unused third-merge scalar before strict loading. Model and optimizer checkpoints are external to this source release.
