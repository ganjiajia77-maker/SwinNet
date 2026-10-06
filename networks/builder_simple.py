# Simplified builder without mmcv dependency
import warnings

class Registry:
    """Simple registry for building modules."""
    
    def __init__(self, name):
        self.name = name
        self._module_dict = {}
    
    def register_module(self, cls=None, force=False):
        """Register a module."""
        if cls is None:
            # Used as decorator with parentheses
            def wrapper(cls):
                self.register_module(cls, force=force)
                return cls
            return wrapper
        
        # Direct class registration
        module_name = cls.__name__
        if module_name in self._module_dict and not force:
            raise ValueError(f'Module {module_name} is already registered in {self.name}')
        
        self._module_dict[module_name] = cls
        return cls
    
    def get(self, name):
        """Get a registered module."""
        if name not in self._module_dict:
            raise KeyError(f'Module {name} is not registered in {self.name}')
        return self._module_dict[name]


BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
SEGMENTORS = Registry('segmentor')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict): The config of modules.
        registry (Registry): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
    """
    if isinstance(cfg, dict):
        cfg = cfg.copy()
        module_name = cfg.pop('type')
        args = cfg
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
        cls = registry.get(module_name)
        return cls(**args)
    else:
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
