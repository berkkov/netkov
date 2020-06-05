import torch


def get_free_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c - a  # free inside cache
    return f'Total: {t:,} || Cached: {c:,} || Allocated {a:,} || {f:,}'


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epochs_per_step=15, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every epochs"""
    # Skip gamma update on first epoch.
    if epoch != 0 and epoch % epochs_per_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
            print("learning rate adjusted: {}".format(param_group['lr']))

