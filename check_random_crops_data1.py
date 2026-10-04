"""Check real DataLoader workers produce changing train crop coordinates."""
import argparse
from types import SimpleNamespace

from torch.utils.data import DataLoader, Subset

from dataloaders.datasets.data1_random512 import Data1Random512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    config = SimpleNamespace(data_root=args.data_root, crop_size=512, base_size=1024,
                             random_crop_train=True, seed=args.seed)
    dataset = Data1Random512(config, split='train')
    subset = Subset(dataset, range(min(args.samples, len(dataset))))
    positions = []
    for epoch in (0, 1, 2):
        dataset.set_epoch(epoch)
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=args.workers,
                            persistent_workers=False)
        samples = {}
        for batch in loader:
            index = int(batch['crop_index'][0])
            samples[index] = (int(batch['crop_top'][0]), int(batch['crop_left'][0]))
        positions.append(samples)
        print('epoch {}: {}'.format(epoch + 1, samples), flush=True)
    changed = sum(len({row[index] for row in positions}) > 1 for index in positions[0])
    if not changed:
        raise RuntimeError('All inspected images reused the same crop across three epochs')
    print('PASS: {}/{} inspected images changed crop coordinates'.format(changed, len(positions[0])))


if __name__ == '__main__':
    main()
