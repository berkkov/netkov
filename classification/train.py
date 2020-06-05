import torch
import torch.nn as nn
from itertools import chain
from classification.loss import NormSoftmaxLoss
from classification.network import NetkovSOTA
from classification.data_loader import TrainSampler, TrainLoader, TestLoader
from classification.metrics import metrics_at_k
from classification.utils import Metric, adjust_learning_rate
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import random


seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

suffix = datetime.datetime.today().strftime("%d_%m_%Y_%H_%M")
CHECKPOINT = None
checkpoint_prefix = 'netkov_v3_run5'
iterations = 1
writer = SummaryWriter(log_dir=f'{checkpoint_prefix}_logs', filename_suffix=suffix, flush_secs=20)
embedding_size = 1024
K_LIST = [1, 3, 5, 8, 10, 15, 20]
MAX_EPOCH = 500
NUM_WARM_UP_EPOCHS = -2
TRAIN_NUM_SAMPLES_PER_CLASS = 4
TRAIN_BATCH_SIZE = 48
weight_decay_gamma = 0.1

img_to_proxy = torch.load('C:\\Users\\user\\Desktop\\anno\\amazon_train_img_to_label.pkl')
num_classes = len(set(img_to_proxy.values()))

optimizer_params = {
    'weight_decay' : 1e-4,
    'momentum' : 0.9,
    'learning_rate' : 1e-2,
}
model = NetkovSOTA(embedding_size=embedding_size).cuda()

loss_function = NormSoftmaxLoss(embedding_size, num_classes)

optimizer = torch.optim.SGD(chain(model.parameters(), loss_function.parameters()),
                            lr=optimizer_params['learning_rate'],
                            momentum=optimizer_params['momentum'],
                            weight_decay=optimizer_params['weight_decay'])

pretrain_optimizer = torch.optim.SGD(chain(list(set(model.parameters()) - set(model.backbone.parameters())),
                                           loss_function.parameters()),
                                     lr=optimizer_params['learning_rate'],
                                     momentum=optimizer_params['momentum'],
                                     weight_decay=optimizer_params['weight_decay'])

track_train_loss = Metric()

sampler = TrainSampler(img_to_proxy_path='C:\\Users\\user\\Desktop\\anno\\amazon_train_img_to_label.pkl',
                       batch_size=TRAIN_BATCH_SIZE,
                       num_samples_per_class=TRAIN_NUM_SAMPLES_PER_CLASS,
                       replacement=False,
                       replacement_labels=False)

train_loader = torch.utils.data.DataLoader(TrainLoader('C:\\Users\\user\\Desktop\\anno\\amazon_train_img_to_label.pkl',
                                                       model.input_config,
                                                       path_prefix='C:\\Users\\user\\Desktop\\netkov_data\\all\\'),
                                           batch_sampler=sampler,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(
        TestLoader('C:\\Users\\user\\Desktop\\anno\\amazon_test_img_to_label.pkl',
                   model.input_config,
                   path_prefix='C:\\Users\\user\\Desktop\\netkov_data\\all\\'),
        batch_size=16,
        shuffle=False,
        num_workers=0)


qap_query_test_loader = torch.utils.data.DataLoader(
    TestLoader('C:\\Users\\user\\Desktop\\anno\\qap_query_to_catalog.pkl',
               model.input_config,
               path_prefix='C:\\Users\\user\\Desktop\\netkov_data_2\\'),
    batch_size=16,
    shuffle=False,
    num_workers=0)

qap_catalog_test_loader = torch.utils.data.DataLoader(
    TestLoader('C:\\Users\\user\\Desktop\\anno\\qap_catalog_path_to_id.pkl',
               model.input_config,
               path_prefix='C:\\Users\\user\\Desktop\\netkov_data_2\\',
               catalog_crowders_to_proxy_path='C:\\Users\\user\\Desktop\\anno\\crowder_img_path_to_name.pkl'),
    batch_size=16,
    shuffle=False,
    num_workers=0)

# If a checkpoint is provided, load the checkpoint
if CHECKPOINT:
    print("=> loading checkpoint '{}'".format(CHECKPOINT))
    checkpoint = torch.load(CHECKPOINT)
    START_EPOCH = checkpoint['epoch'] + 1
    print("Epoch: ", START_EPOCH)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_function.load_state_dict(checkpoint['loss_function'])
    iterations = checkpoint['iterations']
    print("Iterations: {}".format(iterations))
    for param_group in optimizer.param_groups:
        print('Current Learning Rate: ', param_group['lr'])
        param_group['lr'] = 1e-5

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(CHECKPOINT, checkpoint['epoch']))
    del checkpoint
else:
    START_EPOCH = 1


# ##########
# TRAINING #
# ##########
model.train()
for epoch in range(START_EPOCH, MAX_EPOCH + 1):
    track_train_loss = Metric()
    if epoch <= NUM_WARM_UP_EPOCHS:

        for batch_id, (x, y) in enumerate(train_loader):

            pretrain_optimizer.zero_grad()
            # Forward pass
            fvs = model(x.cuda())

            # Calculate loss
            loss = loss_function(fvs.cuda(), y.cuda())
            track_train_loss.update(loss.item(), len(x))
            writer.add_scalar('Training Latest Loss', track_train_loss.val, iterations)
            writer.add_scalar('Training Average Loss', track_train_loss.avg, iterations)

            # Gradient descent
            loss.backward()
            pretrain_optimizer.step()
            pretrain_optimizer.zero_grad()
            iterations += 1

            print(
                f'\r(Pre-training) Epoch: {epoch} || {batch_id + 1} / {len(train_loader)} '
                f'|| Average Loss: {track_train_loss.avg:.4f}',
                end='')
        if epoch == NUM_WARM_UP_EPOCHS:
            del pretrain_optimizer

    else:
        for batch_id, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            fvs = model(x.cuda())

            # Calculate loss
            loss = loss_function(fvs.cuda(), y.cuda())
            track_train_loss.update(loss.item(), len(x))
            writer.add_scalar('Training Latest Loss', track_train_loss.val, iterations)
            writer.add_scalar('Training Average Loss', track_train_loss.avg, iterations)

            # Gradient descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iterations += 1

            print(f'\rEpoch: {epoch} || {batch_id + 1} / {len(train_loader)} || Average Loss: {track_train_loss.avg:.4f}',
                  end='')

    print('')

    # Plot training loss after epoch and create a checkpoint of model state and loss
    writer.add_scalar('Train Loss at Epoch', track_train_loss.avg, epoch)

    # Plot learning rate as well
    _lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Leaning Rate at Epoch', _lr, epoch)

    if epoch % 5 == 0:
        _checkpoint = \
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_function': loss_function.state_dict(),
                'loss_avg': track_train_loss.avg,
                'losses_count': track_train_loss.count,
                'losses_sum': track_train_loss.sum
            }
        torch.save(_checkpoint, f'{checkpoint_prefix}_{epoch}.pkl')

    if epoch % 25 == 0 and _lr > 1e-3:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= weight_decay_gamma

    if epoch % 100 == 0 and _lr > 1e-4:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.50

    # ###############
    # ###############
    # Test 1 - Amazon
    # ###############
    # ###############

    if epoch % 100 == 0:
        # Process test images
        model.eval()
        eval_paths = []
        eval_fvs = []
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(test_loader):
                print(f'\r{batch_id} / {len(test_loader) - 1}', end='')
                x = x.cuda()
                fvs = model(x)
                fvs = fvs.cpu()
                eval_fvs.append(fvs)
                eval_paths.extend(y)
                del fvs, x, y

        # Concatenate all feature vectors to get N x D matrix where N is the number of images and D is the embedding size
        eval_fvs = torch.cat(eval_fvs, dim=0)

        # Calculate cosine similarity. This is an efficient formula since the feature vectors are normalized to have unit norm.
        co_sim_matrix = torch.matmul(eval_fvs,eval_fvs.T)

        # Sort the similarity matrix and also the indices accordingly to evaluate recall and catch at K.
        sorted_co_sim_matrix, sorted_indices = torch.sort(co_sim_matrix, descending=True)

        # Get the test annotations and convert img-to-label to img-to-similar
        amazon_test_img_to_label = torch.load('C:\\Users\\user\\Desktop\\anno\\amazon_test_img_to_label.pkl')
        amazon_test_img_index_to_label = {eval_paths.index(k): v for k, v in amazon_test_img_to_label.items()}
        amazon_test_img_index_to_catalog = {k: [z for z in amazon_test_img_index_to_label
                                                if amazon_test_img_index_to_label[z]==v if z!=k]
                                            for k, v in amazon_test_img_index_to_label.items()}

        # Calculate metrics at k
        for k in K_LIST:
            results = metrics_at_k(sorted_indices, k, amazon_test_img_index_to_catalog)
            print(f'Amazon test results for K={k} '
                  f'|| Adj-Recall: {results[0]:.4f} '
                  f'|| Recall: {results[1]:.4f} '
                  f'|| Catch: {results[2]:.4f}')
            writer.add_scalar(f'Amazon Test || Adj Recall at {k}', results[0], epoch)
            writer.add_scalar(f'Amazon Test || Recall at {k}', results[1], epoch)
            writer.add_scalar(f'Amazon Test || Catch at {k}', results[2], epoch)

        del eval_paths, eval_fvs

        # ############
        # ############
        # Test 2 - qap
        # ############
        # ############

        # First, process query images
        eval_query_paths = []
        eval_query_fvs = []
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(qap_query_test_loader):
                print(f'\rCalculating FVs for QAP anchors {batch_id} / {len(qap_query_test_loader) - 1}', end='')
                x = x.cuda()
                fvs = model(x)
                fvs = fvs.cpu()
                eval_query_fvs.append(fvs)
                eval_query_paths.extend(y)
                del fvs, x, y

        eval_query_fvs = torch.cat(eval_query_fvs)

        # Then, process Catalog images
        eval_catalog_paths = []
        eval_catalog_fvs = []
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(qap_catalog_test_loader):
                print(f'\rCalculating FVs for QAP catalog {batch_id} / {len(qap_catalog_test_loader) - 1}', end='')
                x = x.cuda()
                fvs = model(x)
                fvs = fvs.cpu()
                eval_catalog_fvs.append(fvs)
                eval_catalog_paths.extend(y)
                del fvs, x, y

        eval_catalog_fvs = torch.cat(eval_catalog_fvs)

        # Calculate cosine similarity. This is an efficient formula since the feature vectors are normalized to have unit
        # norm.
        co_sim_matrix = torch.matmul(eval_query_fvs, eval_catalog_fvs.T)

        # Sort the similarity matrix and also the indices accordingly to evaluate recall and catch at K.
        sorted_co_sim_matrix, sorted_indices = torch.sort(co_sim_matrix, descending=True)

        qap_query_to_catalog = torch.load('C:\\Users\\user\\Desktop\\anno\\qap_query_to_catalog.pkl')
        qap_query_to_catalog_indices = {eval_query_paths.index(q): [eval_catalog_paths.index(v)
                                                                    for v in qap_query_to_catalog[q]]
                                        for q in qap_query_to_catalog.keys()}

        # Calculate metrics at K
        for k in K_LIST:
            results = metrics_at_k(sorted_indices, k, qap_query_to_catalog_indices)
            print(f'QAP test results for K={k} '
                  f'|| Adj-Recall: {results[0]:.4f} '
                  f'|| Recall: {results[1]:.4f} '
                  f'|| Catch: {results[2]:.4f}')
            writer.add_scalar(f'QAP Test || Adj Recall at {k}', results[0], epoch)
            writer.add_scalar(f'QAP Test || Recall at {k}', results[1], epoch)
            writer.add_scalar(f'QAP Test || Catch at {k}', results[2], epoch)

        del eval_query_paths, eval_catalog_paths, eval_query_fvs, eval_catalog_fvs
