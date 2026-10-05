OPTION_DEFAULTS = dict(neighbours=4, max_distance=64.0, samples=8,
                       corridor_offset=2.0, hidden=64, prior_warmup_epochs=10,
                       prior_ramp_epochs=5)


def add_anchor_options(parser):
    parser.add_argument('--global_topology_mode', choices=('feature_anchors', 'supervised_anchors'),
                        default='feature_anchors')
    for name, value in OPTION_DEFAULTS.items():
        parser.add_argument('--anchor_' + name, type=type(value), default=value)


def anchor_options(args):
    return {name: getattr(args, 'anchor_' + name) for name in OPTION_DEFAULTS}


def restore_anchor_options(args, saved_args, cli):
    for name in ['global_topology_mode'] + ['anchor_' + key for key in OPTION_DEFAULTS]:
        if name in saved_args and not any(token == '--' + name or token.startswith('--' + name + '=')
                                           for token in cli):
            setattr(args, name, saved_args[name])
