import torch
from ProxyUtils import ProxyLoss, ProxyDataLoader, recall_test, ProxyTestLoader
from Netkov import Netkov
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import os
from NetworkUtils import Metric
torch.cuda.CUDA_LAUNCH_BLOCKING = 1

RESUME = "weights/proxy_checkpoint_4_371.pth.tar"
LOG_INTERVAL = 1
writer = SummaryWriter(log_dir='proxy\\run_adam4cont_0_01')
global iterations
global test_iterations

iterations = 0
test_iterations = 0
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 100
all_dir = "D:\\all_dresses"
cudnn.benchmark = True

# Train information files
all_to_proxy = torch.load("proxy logs\\all_to_proxy.pkl")
proxy_dic = torch.load("proxy logs\\proxy_dic.pkl")

# Test information files
extra_cat_ids = torch.load("proxy logs\\test_extra_cat_ids.pkl")
truth = torch.load("proxy logs\\test_anc_to_pos.pkl")

# Macro Parameters
num_proxy = len(proxy_dic)
EMBEDDING_SIZE = 128


train_loader = torch.utils.data.DataLoader(ProxyDataLoader(all_dir=all_dir,
                                                           all_to_proxy=all_to_proxy,
                                                           network_type='inception'),
                                           batch_size=TRAIN_BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(ProxyTestLoader(all_dir=all_dir,
                                                          test_anc_to_cat=truth,
                                                          extra_cat_ids=extra_cat_ids,
                                                          network_type='inception'),
                                          batch_size=TEST_BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=0)




proxies = torch.nn.Embedding(num_proxy, EMBEDDING_SIZE).cuda()
torch.nn.init.xavier_uniform_(proxies.weight)
model = Netkov(include_shallow_net=False, backbone='inception', embedding_size=EMBEDDING_SIZE)
model.cuda()

for i in model.backbone.children():
    for j in i.parameters():
        print(j.requires_grad)


count = 0
for i in model.backbone.features.children():
    if count >= 15:
        for j in i.parameters():
            j.requires_grad = True

    count += 1

print(count)

for i in model.backbone.children():
    for j in i.parameters():
        print(j.requires_grad)


for i in list(model.state_dict().keys()):
    print(i)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-8)
#optimizer = torch.optim.RMSprop(model.parameters(),
loss_function = ProxyLoss(proxies, proxy_dic, all_to_proxy, writer=writer).cuda()






start_epoch = 1
if RESUME:
    if os.path.isfile(RESUME):
        print("=> loading checkpoint '{}'".format(RESUME))
        checkpoint = torch.load(RESUME)
        start_epoch = checkpoint['epoch'] + 1
        print("Epoch: ", start_epoch)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_function.load_state_dict(checkpoint['proxies'])

        #for param_group in optimizer.param_groups:
        #    print(param_group['lr'])
        #    param_group['lr'] = 0.001
        #    param_group['eps'] = 1e-5
        #del param_group

        if checkpoint:
            metric_cache = {
                'losses_avg': checkpoint['loss_avg'], 'losses_sum': checkpoint['losses_sum'],
                'losses_count': checkpoint['losses_count'],
            }

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(RESUME, checkpoint['epoch']))

        del checkpoint


def train(model, train_loader, loss_function, optimizer, epoch_no):
    print("***Training begins***")
    losses = Metric()
    p_dist_metric = Metric()
    model.train()
    global iterations
    for batch_id, (q, q_id) in enumerate(train_loader):

        iterations += 1
        q = model(q)

        optimizer.zero_grad()
        loss = loss_function(q, q_id, batch_size=len(q), p_dist_metric=p_dist_metric)
        losses.update(loss.item(), len(q))
        loss.backward()
        optimizer.step()

        if batch_id % 20 == 0:
            print('Training: {} / {}\t'
                  'Mean Loss: {:.4f}'
                  .format(batch_id, len(train_loader), losses.avg))

        writer.add_scalar("Training Average Loss", losses.avg, iterations)
        writer.add_scalar('Training Latest Loss', losses.val, iterations)

    if epoch_no % LOG_INTERVAL == 0:
        print('Train Epoch: {} [{}/{}]\t'
              'Loss: {:.4f} mean({:.4f}) currentm({:.4f}) currentm({})'
              .format(epoch_no, (batch_id + 1) * len(q), len(train_loader.dataset),
                      losses.sum, losses.avg, loss, losses.val))

        filename = 'proxy_checkpoint_' + str(epoch_no) + "_" + str(batch_id + 1) + ".pth.tar"

        save_checkpoint({
            'epoch': epoch_no,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_avg': losses.avg,
            'losses_count': losses.count,
            'losses_sum': losses.sum,
            'proxies': loss_function.state_dict(),
        }, filename=filename)

    del q
    optimizer.zero_grad()
    writer.add_scalar("Train Avg Loss v Batch", losses.avg, epoch_no)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "weights"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)


def test(model, test_loader, epoch_no):
    print("***Evaluation begins***")
    model.eval()
    evaluated_tensors = []
    evaluated_ids = []


    with torch.no_grad():
        for batch_id, (q, q_id) in enumerate(test_loader):
            print("Test Images Evaluation Progress: %{:.2f}".format((batch_id + 1) / len(test_loader) * 100))
            q = model(q)
            evaluated_tensors.append(q)
            evaluated_ids.append(q_id)

        fvs = torch.cat(evaluated_tensors, dim=0)
        del evaluated_tensors
        ids = torch.cat(evaluated_ids)
        del evaluated_ids, q, q_id

    result = recall_test(fvs, ids, truth)
    for k in result:
        recall = result[k]
        writer.add_scalar("Test Recall at " + str(k), recall, epoch_no)

    return


for epoch in range(start_epoch, 20):
    #train(model=model, train_loader=train_loader, loss_function=loss_function, optimizer=optimizer, epoch_no=epoch)
    #test(model, test_loader, epoch)
    pass