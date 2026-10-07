import argparse


def add_encoder_arguments(parser):
    parser.add_argument("--encoder_type", choices=("swin", "dinov2_l16"), default=None)
    parser.add_argument("--freeze_pretrained_encoder", action=argparse.BooleanOptionalAction, default=None)


def inherit_encoder_arguments(args, checkpoint):
    saved = checkpoint.get("args", {})
    state = checkpoint.get("training_model_state_dict", checkpoint.get("model_state_dict", {}))
    actual = "dinov2_l16" if any("dino_encoder.backbone." in key for key in state) else "swin"
    recorded = saved.get("encoder_type", actual)
    if recorded != actual:
        raise ValueError(f"Checkpoint encoder metadata {recorded} disagrees with tensors {actual}")
    if args.encoder_type is not None and args.encoder_type != actual:
        raise ValueError(f"Cannot resume {actual} checkpoint with {args.encoder_type} encoder")
    args.encoder_type = actual
    if args.freeze_pretrained_encoder is None:
        args.freeze_pretrained_encoder = bool(saved.get("freeze_pretrained_encoder", actual == "dinov2_l16"))
