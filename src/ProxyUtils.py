import torch
from torch.autograd import Variable
import torch.utils.data
from NetworkUtils import LoadImage, TransformImage
import os
import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

torch.cuda.CUDA_LAUNCH_BLOCKING=1


class ProxyLoss(torch.nn.Module):
    def __init__(self, proxies, proxy_dic, anc_to_proxy, writer):
        super(ProxyLoss, self).__init__()
        self.proxy_dic = proxy_dic
        self.anc_to_proxy = anc_to_proxy
        self.writer = writer

        self.num_proxy = proxies.weight.size(0)

        self.proxies = proxies
        self.dist = torch.nn.PairwiseDistance(eps=1e-16)

    def nca(self, q_embeds, q_ids, anc_no, p_dist_metric):
        q = q_embeds[anc_no]

        proxy_id = torch.Tensor([self.anc_to_proxy[str(q_ids[anc_no].item())]]).cuda().long()

        p = self.proxies(proxy_id)
        p = F.normalize(p, dim=1)

        N_ids = torch.masked_select(
            Variable(
                torch.arange(0, self.num_proxy).long()
            ).cuda(),
            Variable(
                torch.arange(0, self.num_proxy).long()
            ).cuda() != proxy_id
        ).long()

        assert N_ids.size(0) == self.num_proxy - 1
        N = self.proxies(N_ids)
        N = F.normalize(N, dim=1)

        p_dist = self.dist(q, p)
        p_dist_metric.update(p_dist, 1)
        self.writer.add_scalar("Training Avg Pos Distance", p_dist_metric.avg, p_dist_metric.count)

        p_dist = torch.exp(-p_dist)
        N_dist = torch.exp(-self.dist(q.expand(N.size(0), q.size(0)), N))
        nca_loss = -torch.log(p_dist / torch.sum(N_dist))
        return nca_loss

    def forward(self, q_embeds, q_ids, batch_size, p_dist_metric):
        loss = torch.mean(torch.stack([self.nca(q_embeds, q_ids, anc_no, p_dist_metric) for anc_no in range(batch_size)]))
        return loss


class ProxyDataLoader(torch.utils.data.Dataset):
    def __init__(self, all_dir, all_to_proxy, network_type):
        super(ProxyDataLoader).__init__()
        self.all_dir = all_dir
        self.loader = LoadImage()
        self.anc_ids = list(all_to_proxy.keys())
        self.transformer = TransformImage(network_type)
        print("Training pool size: ", len(self.anc_ids))

    def __getitem__(self, index):
        q = torch.autograd.Variable(self.transformer(
            self.loader(os.path.join(self.all_dir, self.anc_ids[index] + ".jpg"))
        )).cuda()
        q_id = self.anc_ids[index]
        return q, int(q_id)

    def __len__(self):
        return len(self.anc_ids)


class ProxyTestLoader(torch.utils.data.Dataset):
    def __init__(self, all_dir, test_anc_to_cat, extra_cat_ids, network_type):
        super(ProxyTestLoader).__init__()
        self.all_dir = all_dir
        self.loader = LoadImage()
        self.transformer = TransformImage(network_type)

        self.query_ids = list(test_anc_to_cat.keys())
        self.extra_cat_ids = extra_cat_ids
        self.cat_ids = []
        for anc in test_anc_to_cat:
            for pos in test_anc_to_cat[anc]:
                if pos not in self.cat_ids:
                    self.cat_ids.append(pos)

        print("Positive Catalog Length: ", len(self.cat_ids))
        self.image_pool = self.query_ids + self.cat_ids + self.extra_cat_ids
        print("Total Catalog Length: ", len(self.cat_ids) + len(self.extra_cat_ids))
        print("Total Image Pool Length: ", len(self.image_pool))

    def __getitem__(self, index):
        q = torch.autograd.Variable(self.transformer(
            self.loader(os.path.join(self.all_dir, self.image_pool[index] + ".jpg"))
        )).cuda()
        q_id = self.image_pool[index]
        return q, int(q_id)

    def __len__(self):
        return len(self.image_pool)


def recall_test(fvs, q_ids, truth):
    """

    :param X: Output embeddings from model
    :param T: Target catalog ids
    :param k: K
    :return:
    """
    result_dic = {}
    max_k = 10
    k_list = [1, 3, 5, 8, 10]
    size = len(truth)

    print("Calculating distances")
    dists = pairwise_distances(fvs, metric='l2')
    indices = np.argsort(dists, axis=1)[:, 1: max_k + 1]
    predictions = np.array([[q_ids[i] for i in ii] for ii in indices])
    q_ids = q_ids.tolist()

    test_anchors = list(truth.keys())

    for k in k_list:
        sanity_check = 0
        corrects = 0
        zipped = zip(q_ids, predictions)
        for id, pred in zipped:
            if str(id) in test_anchors:
                sanity_check += 1
                positives = truth[str(id)]

                for pos in positives:
                    if int(pos) in pred[:k]:
                        corrects += 1
                        break

        if sanity_check != size:
            print(sanity_check)
            print(size)
            print("Recall Pool Length Sanity Check Failed at k={}".format(k))
        result_dic[k] = (corrects/size)*100

    print(result_dic)
    return result_dic


"""
t = torch.load("proxy logs\\test_anc_to_pos.pkl")
r = torch.load("proxy logs\\u3_anc_pos.pkl")

count =0
forbidden = []
for a in t:
    for p in t[a]:
        if p not in forbidden:
            forbidden.append(p)

for a in r:
    for p in r[a]:
        if p not in forbidden:
         forbidden.append(p)

import glob
cat = glob.glob("D:\\netkov\\data\\structured\\dresses_catalog\\*.jpg")
print(len(cat))

print(len(forbidden))
cat = [x.split('\\')[-1].split('.')[0] for x in cat]
print(cat)
extra_cat_ids = []
for i in cat:
    if i not in forbidden:
        extra_cat_ids.append(i)
        count += 1
        if count == 10000-2269:
            break

torch.save(extra_cat_ids, "proxy logs\\test_extra_cat_ids.pkl")
"""
