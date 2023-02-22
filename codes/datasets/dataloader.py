import torch
import os
from torch.utils.data import dataloader, distributed, DataLoader
from datasets.yolov5_dataset import Yolov5Dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def create_dataloader(path, is_train=True, img_size=640, batch_size=8, workers=8, rank=-1,
                      shuffle=True):
    dataset = Yolov5Dataset(path, is_train=True, mosaic=True, img_size=img_size)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw, collate_fn=dataset.collate_fn)

    return loader, dataset

