import torch
from Netkov import Netkov, TripletNetkov
import torch.backends.cudnn as cudnn
from NetworkUtils import TripletLoader, Metric, ContinuedSampler
import torch.utils.data
from datetime import datetime
import os
import argparse
from tensorboard_logger import configure, log_value
from OnlineSampler import HardTripletSampler
import random
from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser(description='Netkov Model Trainer Script')

parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 2)')

parser.add_argument('--test-batch-size', type=int, default=16, metavar='T',
                    help='input batch size for testing (default: 16)')

parser.add_argument('--log-interval', type=int, default=5, metavar='L',
                    help='how many iterations to wait before logging training status')

parser.add_argument('--resume', default=None, type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--sequence-file', default=None, type=str,
                    help='path to latest sequence file if the model saved inside an epoch')

parser.add_argument('--max-epoch', default=40000, type=int, metavar='E',
                    help='Max number of epochs (default: 10)')

parser.add_argument('--margin', type=float, default=0.40, metavar='M',
                    help='margin for triplet loss (default: 5)')

parser.add_argument('--category', default='', type=str, metavar='C',
                    help='Category of the model')

parser.add_argument('--pathtocrop', default="/content/data/structured/wtbi_dresses_query_crop", type=str, metavar='PCR',
                    help='Path to dresses_catalog and to crop dir')

parser.add_argument('--pathtocat', default="/content/data/structured/dresses_catalog", type=str, metavar='PCT',
                    help='Path to dresses_catalog and to crop dir')

parser.add_argument('--runbatchsize', default=120, type=int, metavar='RBS')

args = parser.parse_args()

configure("logs", flush_secs=5)

best_acc = 0

TRAIN_BATCH_SIZE = args.batch_size # args.batch_size
TEST_BATCH_SIZE = 48  # args.test_batch_size
LOG_INTERVAL = args.log_interval  # args.log_interval
MAX_EPOCH = args.max_epoch  # args.max_epoch + 1
MARGIN = args.margin  # args.margin
RESUME = args.resume    # 'weights/clear_checkpoint_73_241.pth.tar'  # args.resume
PATH_TO_CROP = args.pathtocrop
PATH_TO_CAT = args.pathtocat
SEQUENCE_PATH = args.sequence_file  # args.sequence_file
RUN_BATCH_SIZE = args.runbatchsize      #For online sampling
global iterations
global test_iterations

print(RESUME)
# Changeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
iterations = 0
test_iterations = 0


def main():
    train_triplets_file = "hardtrain.csv"  # TODO : Change - DONE FOR DRIVE
    test_triplets_file = "test_random.csv"  # TODO : Change - DONE FOR DRIVE
    global args, best_acc, iterations, test_iterations
    cudnn.benchmark = True
    model = TripletNetkov()
    model.cuda()
    model.train()
    loss_function = torch.nn.TripletMarginLoss(margin=MARGIN, reduce=False)  # TODO FILL PARAMETERS _ DONE FOR DRIVE v0

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # TODO FILL PARAMETERS
    optimizer = torch.optim.Adadelta(model.parameters())

    #test_loader = torch.utils.data.DataLoader(TripletLoader(test_triplets_file), batch_size=TEST_BATCH_SIZE,
     #                                         shuffle=False,
      #                                        num_workers=0)

    loader_check = True  # This is to determine which loader to use, if True we use the clean loader, if False
    # continued loader is used
    start_epoch = 0
    if RESUME:
        if os.path.isfile(RESUME):
            print("=> loading checkpoint '{}'".format(RESUME))
            checkpoint = torch.load(RESUME)
            start_epoch = checkpoint['epoch'] + 1
            print("Epoch: ", start_epoch)
            best_acc = checkpoint['best_prec']
            print(model.state_dict().keys())
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            last_batch_id = checkpoint['last_batch_id']
            # iterations = checkpoint['iterations']
            print("Iterations: ", iterations)
            if checkpoint:
                metric_cache = {
                    'losses_avg': checkpoint['loss_avg'], 'losses_sum': checkpoint['losses_sum'],
                    'losses_count': checkpoint['losses_count'], 'acc_avg': checkpoint['acc_avg'],
                    'acc_sum': checkpoint['acc_avg'], 'acc_count': checkpoint['acc_count']
                }

            print("=> loaded checkpoint '{}' (epoch {}) (iteration {})"
                  .format(RESUME, checkpoint['epoch'], checkpoint['last_batch_id']))

            if SEQUENCE_PATH and last_batch_id:
                print('Last batch id was ', last_batch_id)
                sampler = ContinuedSampler(TripletLoader(train_triplets_file), SEQUENCE_PATH, last_batch_id,
                                           TRAIN_BATCH_SIZE)
                train_loader = torch.utils.data.DataLoader(TripletLoader(train_triplets_file),
                                                           batch_size=TRAIN_BATCH_SIZE,
                                                           shuffle=False, sampler=sampler, num_workers=0)
                loader_check = False

            del checkpoint

        else:
            print("=> no checkpoint found at '{}'".format(RESUME))

    for epoch in range(start_epoch, MAX_EPOCH):

        if loader_check:
            pass
            print("Default trainer is active")
            if epoch != 11241241:
                seed = random.randint(0,10000)
                torch.cuda.manual_seed(seed)
                online_sampler = HardTripletSampler(model, path_to_crops=PATH_TO_CROP, path_to_catalog=PATH_TO_CAT)
                online_sampler.sample_fast(num_anchors=160, run_batch_size=RUN_BATCH_SIZE, extra_cat_num=2000)
                del online_sampler
                train_triplets_file = "hardtrain.csv"
                train_loader = torch.utils.data.DataLoader(TripletLoader(train_triplets_file), batch_size=TRAIN_BATCH_SIZE,
                                                           shuffle=True, num_workers=0)

                train_netkov(model=model, train_loader=train_loader, loss_function=loss_function, optimizer=optimizer,
                             epoch_no=epoch)
        else:
            train_netkov(model=model, train_loader=train_loader, loss_function=loss_function, optimizer=optimizer,
                         epoch_no=epoch, metric_cache=metric_cache, last_batch=last_batch_id)
            pass
        # with torch.no_grad():
            # test_netkov(model=model, test_loader=test_loader, loss_function=loss_function, epoch_no=epoch)

        loader_check = True


def train_netkov(model, train_loader, loss_function, optimizer, epoch_no, log_interval=LOG_INTERVAL,
                 metric_cache=None, last_batch=None):
    global best_acc, iterations
    losses = Metric()
    accuracies = Metric()
    if metric_cache:
        losses.avg = metric_cache['losses_avg']
        losses.sum = metric_cache['losses_sum']
        losses.count = metric_cache['losses_count']
        accuracies.avg = metric_cache['acc_avg']
        accuracies.sum = metric_cache['acc_sum']
        accuracies.count = metric_cache['acc_count']

    model.train()
    count = 0
    # print("Giris,", datetime.now())
    for batch_id, (q, p, n) in enumerate(train_loader):
        # print("Batch giris, ", datetime.now())
        if last_batch:
            if batch_id <= last_batch:
                if batch_id % 500 == 0:
                    print('Skipping batch ', batch_id)
                continue
        if (batch_id % 50) == 49 or batch_id < 50:
            print(str(batch_id + 1) + "/" + str(len(train_loader)), datetime.now())  # DTODO: Remove it - DONE FOR DRIVE
        q = q.cuda()
        p = p.cuda()
        n = n.cuda()
        # print("Model öncesi, ", datetime.now())
        # Forward pass
        q, p, n = model(query=q, positive=p, negative=n)

        # print("Positive dist, " , torch.dist(q[0],p[0]))
        # print("Negative dist, " , torch.dist(q[0],n[0]))
        # print("Model sonrası, ", datetime.now())

        loss = loss_function(q, p, n)
        print(loss)
        # Metric calculations
        acc = torch.sum(loss.data <= MARGIN).item() / len(q)
        fraction_non_zero_losses = torch.sum(loss.data != 0).item() / len(q)
        # print(len(q), fraction_non_zero_losses, torch.sum(loss.data != 0).item())
        # print(torch.sum(loss.data != 0))
        # print(loss)
        loss = torch.mean(loss)

        # Metric updates

        losses.update(loss.item(), len(q))

        accuracies.update(acc, len(q))

        # print("optim öncesi ", datetime.now())
        # Optimization steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("optim sonrası ", datetime.now())
        log_value('Training Average Loss', losses.avg, iterations)
        log_value('Training Latest Loss', losses.val, iterations)
        log_value('Training Average Accuracy', accuracies.avg, iterations)
        log_value('Training Latest Accuracy', accuracies.val, iterations)
        log_value("Train % Non Zero Losses", fraction_non_zero_losses, iterations)

        # iterations = (epoch_no - 1)* len(train_loader) + batch_id
        iterations += 1

        # print("loglar sonrası ", datetime.now())

        if batch_id % log_interval == log_interval - 1 or (epoch_no % 15 == 0 and batch_id == len(train_loader) - 1):
            count += 1
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} mean({:.4f}) currentm({:.4f}) currentm({}) \t'
                  'Acc: current({:.4f}) mean({:.4f})\t'
                  'Time: {}'.format(epoch_no, (batch_id + 1) * len(q), len(train_loader.dataset),
                                    losses.sum, losses.avg, loss, losses.val,
                                    accuracies.val, accuracies.avg, datetime.now()))

            if count % 3 == 0 or batch_id == len(train_loader) - 1:
                is_best = acc > best_acc
                best_acc = max(acc, best_acc)
                filename = 'clear_checkpoint_' + str(epoch_no) + "_" + str(batch_id + 1) + ".pth.tar"
                sequence = {}
                for seq, id in enumerate(train_loader.sampler):
                    sequence[seq] = id  # TODO: Lift Save

                torch.save(sequence, "sequence.pth")  # Save the sampler
                save_checkpoint({
                    'epoch': epoch_no,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec': best_acc,
                    'last_batch_id': batch_id,
                    'loss_avg': losses.avg,
                    'losses_count': losses.count,
                    'losses_sum': losses.sum,
                    'acc_sum': accuracies.sum,
                    'acc_count': accuracies.count,
                    'acc_avg': accuracies.avg,
                    'iterations': iterations
                }, is_best, filename=filename)

        # print("del öncesi ", datetime.now())
        del q, p, n, loss, acc
        optimizer.zero_grad()
        # print("for sonu, ", datetime.now())
    log_value("Train Accuracy v. Batch", accuracies.avg, epoch_no)
    log_value("Train Avg. Loss v. Batch", losses.avg, epoch_no)


def test_netkov(model, test_loader, loss_function, epoch_no):
    print('Testing begins')
    global test_iterations
    losses = Metric()
    accuracies = Metric()

    model.eval()
    count = 0

    for batch_id, (q, p, n) in enumerate(test_loader):
        q = q.cuda()
        p = p.cuda()
        n = n.cuda()

        q, p, n = model(q, p, n)

        # Calculate Metrics
        test_loss = loss_function(q, p, n)
        acc = torch.sum(test_loss <= MARGIN).item() / len(q)

        test_loss = torch.mean(test_loss)
        # Update Metrics
        accuracies.update(acc, len(q))

        losses.update(test_loss, len(q))
        count += 1

        test_iterations += 1

        log_value("Test Average Loss", losses.avg, test_iterations)
        log_value("Test Average Accucy", accuracies.avg, test_iterations)
        log_value("Test Latest Acc", accuracies.val, test_iterations)
        log_value("Test Latest Loss", losses.val, test_iterations)

        if count % LOG_INTERVAL == 0:
            print(
                '\nTest set: Batches Tested: {:.2f}, Average loss: {:.4f}, Accuracy: {:.4f}, L_Accuracy{:.4f}\n'.format(
                    count, losses.avg, accuracies.avg, accuracies.val))

    log_value("Test Accuracy v. Batch", accuracies.avg, epoch_no)
    log_value("Test Avg. Loss v. Batch", losses.avg, epoch_no)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "weights"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    # if is_best:
    # shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == "__main__":
    main()

