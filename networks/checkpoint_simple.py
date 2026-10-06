# Simplified checkpoint loading without mmcv dependency
import torch
import logging
import os

def load_checkpoint(model, filename, strict=False, logger=None):
    """Load checkpoint without mmcv dependency.
    
    Args:
        model: Module to load weights into
        filename (str): Path to checkpoint file
        strict (bool): Whether to strictly enforce matching keys
        logger: Logger for warnings/errors
    """
    if not isinstance(filename, str):
        raise TypeError(f'filename must be a str, but got {type(filename)}')
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    else:
        state_dict = checkpoint
    
    # Load state dict
    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
        loaded_keys = len(state_dict) - len(incompatible.unexpected_keys)
        if loaded_keys == 0:
            raise RuntimeError('No pretrained backbone keys matched: ' + filename)
        logger.info('Loaded %d matching backbone keys from %s', loaded_keys, filename)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            logger.warning(f'Incompatible keys in state_dict: '
                         f'missing {len(incompatible.missing_keys)} keys, '
                         f'unexpected {len(incompatible.unexpected_keys)} keys')
    except Exception as e:
        logger.error(f'Failed to load checkpoint from {filename}: {e}')
        raise
    
    logger.info(f'Loaded checkpoint from {filename}')

