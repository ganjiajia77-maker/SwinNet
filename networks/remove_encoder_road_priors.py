"""Delete A1/A2 and their two encoder uses after identical initialization."""


def remove_encoder_road_priors(model):
    core=model.swin_unet
    if getattr(core,'encoder_road_priors_removed',False):
        raise ValueError('Encoder A1/A2 already removed')
    before=set(core.state_dict())
    for index in (0,1):
        layer=core.layers[index]
        layer.road_attention_head=None
        layer.road_attention_modulates_downsample=False
        layer.last_road_attention=None
        layer.downsample.road_alpha=None
    del core.encoder_stage1_road_attention_head
    del core.encoder_stage2_road_attention_head
    stage3=core.layers[2]
    stage3.use_road_bias=False
    for block in stage3.blocks:
        block.use_road_bias=False
        block.road_bias_scale_a1=None
        block.road_bias_scale_a2=None
    removed=sorted(before-set(core.state_dict()))
    core.encoder_road_priors_removed=True
    core.encoder_road_prior_ablation=dict(
        removed_keys=['swin_unet.'+key for key in removed],
        heads_removed=True,patch_merge_modulation_removed=True,encoder_stage3_bias_removed=True)
    return model
