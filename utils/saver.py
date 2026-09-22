import csv
import glob
import os
import shutil
from collections import OrderedDict
from datetime import datetime

import torch


class Saver(object):
    def __init__(self, args):
        self.args = args
        root = getattr(args, 'output_dir', None) or 'run'
        self.directory = os.path.join(root, args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.directory, 'experiment_' + str(run_id))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        torch.save(state, os.path.join(self.directory, 'last.pth'))
        if not is_best:
            return
        best_pred = state['best_pred']
        with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as handle:
            handle.write(str(best_pred))
        previous_best = []
        for run in self.runs:
            path = os.path.join(run, 'best_pred.txt')
            if os.path.isfile(path):
                with open(path, 'r') as handle:
                    previous_best.append(float(handle.readline()))
        if not previous_best or best_pred > max(previous_best):
            shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            shutil.copyfile(filename, os.path.join(self.directory, 'best.pth'))

    def append_epoch_losses(self, row):
        path = os.path.join(self.directory, 'epoch_losses.csv')
        fields = ['epoch', 'split', 'total_loss', 'loss1', 'loss2', 'loss3',
                  'iou', 'precision', 'recall', 'f1']
        exists = os.path.isfile(path)
        with open(path, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, '') for field in fields})

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        values = OrderedDict()
        values['dataset'] = self.args.dataset
        values['backbone'] = self.args.backbone
        values['out_stride'] = self.args.out_stride
        values['lr'] = self.args.lr
        values['lr_scheduler'] = self.args.lr_scheduler
        values['loss_type'] = self.args.loss_type
        values['epochs'] = self.args.epochs
        values['base_size'] = self.args.base_size
        values['crop_size'] = self.args.crop_size
        with open(logfile, 'w') as handle:
            for key, value in values.items():
                handle.write(key + ':' + str(value) + '\n')