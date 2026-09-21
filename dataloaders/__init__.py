from dataloaders.datasets import spacenet, spacenet_crop, deepglobe, deepglobe_crop
from dataloaders.datasets.data1_random512 import Data1Random512
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def make_data_loader(args, **kwargs):
    if args.dataset == 'data1':
        train_set = Data1Random512(args, split='train')
        val_set = Data1Random512(args, split='val')
        test_set = Data1Random512(args, split='test')
    elif args.dataset == 'spacenet':
        train_set = spacenet_crop.Segmentation(args, split='train')
        val_set = spacenet_crop.Segmentation(args, split='val')
        test_set = spacenet.Segmentation(args, split='test')
    elif args.dataset == 'DeepGlobe':
        train_set = deepglobe_crop.Segmentation(args, split='train')
        val_set = deepglobe_crop.Segmentation(args, split='val')
        test_set = deepglobe.Segmentation(args, split='test')
    else:
        raise NotImplementedError
    return (
        DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs),
        DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs),
        DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs),
        train_set.NUM_CLASSES,
    )