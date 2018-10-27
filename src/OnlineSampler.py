import torch
from torch.nn.functional import triplet_margin_loss
import glob
import random
import NetworkUtils as util
import csv
from Netkov import EvaNetkovCatalog, EvaNetkovQuery


class HardTripletSampler(object):
    def __init__(self, model, path_to_crops, path_to_catalog, train_ids_path="train_ids.pkl",
                 q_to_p_path="anchor_to_positive.pkl", all_pos_path="all_pos.pkl"):
        print("~Hard Triplet Sampler is Active~")
        self.path_to_crops = path_to_crops
        self.path_to_catalog = path_to_catalog
        std = model.state_dict()
        self.shallow_net_status = model.include_shallow_net
        self.backbone_name = model.backbone_name

        self.model_cat = EvaNetkovCatalog(backbone=self.backbone_name, include_shallow_net=self.shallow_net_status)
        self.model_query = EvaNetkovQuery(backbone=self.backbone_name, include_shallow_net=self.shallow_net_status)

        self.model_cat.load_state_dict(std, strict=False)
        self.model_query.load_state_dict(std, strict=False)

        self.anchor_id_list = torch.load(train_ids_path)
        self.query_to_positive = torch.load(q_to_p_path)
        self.all_pos_pth = [path_to_catalog + "/" + pos_id + ".jpg" for pos_id in torch.load(all_pos_path)]
        self.anchor_path_list = [self.path_to_crops + "/" + x + ".jpg" for x in self.anchor_id_list]   # .split("/")[-1].split('.')[0]
        self.catalog_path_list = [x for x in glob.glob(self.path_to_catalog + "/*.jpg")] # .split("/")[-1].split('.')[0]

    def sample(self, num_anchors=300, num_neg_per_pos=10, run_batch_size=32):
        """
        Sampling triplets using the whole catalog.
        :param num_anchors:
        :param num_neg_per_pos:
        :param run_batch_size:
        :return:
        """
        triplets = []
        print("Begin sampling operation")
        anchor_indices = random.sample(range(0, len(self.anchor_path_list)), num_anchors)
        select_anchors = [self.anchor_path_list[j] for j in anchor_indices]
        torch.save(select_anchors, "cache_select_anchors.pkl")
        crop_d = {}
        catalog_d = {}
        if self.backbone_name[0] == 'r':
            loader_type = 'resnet'
        elif self.backbone_name[0] == 'i':
            loader_type = 'inception'

        with torch.no_grad():

            data_loader = torch.utils.data.DataLoader(util.FVBatchLoader(select_anchors,
                                                                         network_type=loader_type),
                                                      batch_size=run_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)
            print("Processing selected anchor images")
            count = 0
            for id, batch in data_loader:
                batch = batch.cuda()
                embeds = self.model_query(batch)
                count += 1
                del batch
                self.model_query.zero_grad()
                if count % 10 == 0 or count == len(data_loader):
                    print(str(count) + "/" + str(len(data_loader)))
                for i in range(0, len(id)):
                    crop_d[id[i]] = embeds[i].cpu()

            data_loader = torch.utils.data.DataLoader(util.FVBatchLoader(glob.glob(self.path_to_catalog + "/*.jpg"),
                                                                         network_type=loader_type),
                                                      batch_size=run_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)
            print("Processing catalog images")
            count = 0
            for id, batch in data_loader:
                batch = batch.cuda()
                embeds = self.model_cat(batch)
                count += 1
                del batch
                self.model_cat.zero_grad()
                if count % 10 == 0 or count == len(data_loader):
                    print(str(count) + "/" + str(len(data_loader)))
                for i in range(0, len(id)):
                    catalog_d[id[i]] = embeds[i].cpu()

            del embeds

            count = 0
            print("Calculating distances")
            anc_to_cat_dist = {}
            for anchor in crop_d:
                anc_to_cat_dist[anchor] = []
                anc_embed = crop_d[anchor]
                count += 1
                if count % 50 == 0:
                    print(count)
                for cat in catalog_d:
                    cat_embed = catalog_d[cat]
                    dist = torch.dist(anc_embed, cat_embed)
                    anc_to_cat_dist[anchor].append((cat, dist.item()))

        for i in anc_to_cat_dist:
            anc_to_cat_dist[i] = sorted(anc_to_cat_dist[i], key=lambda cat_img: cat_img[1])[:50]

        print("Sampling triplets and preparing triplet file")
        for anc in anc_to_cat_dist:
            q = anc

            p_s = []
            for pos in self.query_to_positive[anc]:
                p_s.append(pos)

            for p in p_s:
                previous_n = []                                                 # Below comment / move above
                for j in range(num_neg_per_pos):
                    n = anc_to_cat_dist[anc][j][0]
                    if n not in p_s and n != q and n not in previous_n:         # Consider altering the n's
                        triplets.append((q, p, n))
                        previous_n.append(n)

        with open("hardtrain.csv", "w", newline='') as csvFile:  # base_dir
            writer = csv.writer(csvFile)
            triplets = [[self.path_to_crops + "/" + x[0] + ".jpg", self.path_to_catalog + "/" + x[1] + ".jpg",
                         self.path_to_catalog + "/" + x[2] + ".jpg"] for x in triplets]  # play
            print("Process Completed\nNumber of triplets: ", len(triplets))
            writer.writerows(triplets)
            del triplets

        return

    def sample_fast(self, num_anchors=1000, run_batch_size=150, extra_cat_num=1500, cat_inclusive=False):
        """
        :param num_anchors:
        :param run_batch_size:
        :param extra_cat_num:
        :param cat_inclusive:
        :return:
        """
        triplets = []
        print("Begin sampling operation")
        print("Sampler length: ", len(self.anchor_path_list))
        anchor_indices = random.sample(range(0, len(self.anchor_path_list)), num_anchors)
        select_anchors = [self.anchor_path_list[j] for j in anchor_indices]

        crop_d = {}
        catalog_d = {}
        candidate_positive_ids_u = [self.query_to_positive[anchor.split('/')[-1].split('.')[0]] for anchor in select_anchors]
        candidate_positive_ids = [item for sublist in candidate_positive_ids_u for item in sublist]
        candidate_positive_pth = [self.path_to_catalog + "/" + x + ".jpg" for x in candidate_positive_ids]

        # Extra random catalog images to extend active triplets
        if cat_inclusive:
            extra_cat_universe = [c for c in self.catalog_path_list if c not in candidate_positive_pth]
        else:
            extra_cat_universe = [c for c in self.catalog_path_list if c not in self.all_pos_pth]
        extra_cat_indices = random.sample(range(0, len(extra_cat_universe)), extra_cat_num)
        extra_cats = [extra_cat_universe[ind] for ind in extra_cat_indices]

        # Combine positive catalog images and extras
        candidate_positive_pth += extra_cats
        num_zero = 0

        if self.backbone_name[0] == 'r':
            loader_type = 'resnet'
        elif self.backbone_name[0] == 'i':
            loader_type = 'inception'

        with torch.no_grad():

            data_loader = torch.utils.data.DataLoader(util.FVBatchLoader(select_anchors,
                                                                         network_type=loader_type),
                                                      batch_size=run_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)
            print("Processing selected anchor images")
            count = 0
            for id, batch in data_loader:
                batch = batch.cuda()
                embeds = self.model_query(batch)
                count += 1
                del batch
                self.model_query.zero_grad()
                if count % 10 == 0 or count == len(data_loader):
                    print(str(count) + "/" + str(len(data_loader)))
                for i in range(0, len(id)):
                    crop_d[id[i]] = embeds[i].cpu()

            data_loader = torch.utils.data.DataLoader(util.FVBatchLoader(candidate_positive_pth,
                                                                         network_type=loader_type),
                                                      batch_size=run_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)
            print("Processing catalog images")
            count = 0
            for id, batch in data_loader:
                batch = batch.cuda()
                embeds = self.model_cat(batch)
                count += 1
                del batch
                self.model_cat.zero_grad()
                if count % 10 == 0 or count == len(data_loader):
                    print(str(count) + "/" + str(len(data_loader)))
                for i in range(0, len(id)):
                    catalog_d[id[i]] = embeds[i].cpu()

            del embeds

            n_in_pos = 0
            count = 0
            sanity = 0
            print("Calculating query-to-catalog distances and sampling triplets")
            anc_to_cat_dist = {}
            for anchor in crop_d:
                anc_to_cat_dist[anchor] = []
                anc_embed = crop_d[anchor]
                count += 1
                positive_only = []
                if count % 50 == 0:
                    print(count)
                for cat in catalog_d:
                    cat_embed = catalog_d[cat]
                    dist = torch.dist(anc_embed, cat_embed)
                    anc_to_cat_dist[anchor].append((cat, dist.item()))
                    if cat in self.query_to_positive[anchor]:
                        positive_only.append((cat, dist.item()))

                if not positive_only:
                    continue

                anc_to_cat_dist[anchor] = sorted(anc_to_cat_dist[anchor], key=lambda cat_img: cat_img[1])[:50]
                # Pick the hardest positive
                p = max(positive_only, key=lambda t: t[1])
                # Pick the hardest negative
                n = anc_to_cat_dist[anchor][0][0]
                n_next = 1

                while n in self.query_to_positive[anchor]:
                    n = anc_to_cat_dist[anchor][n_next][0]
                    n_next += 1
                    if n_next == 5:
                        break

                if n not in self.query_to_positive[anchor]:
                    while sanity < 1:
                        print(crop_d[anchor].unsqueeze(0))
                        sanity += 1
                    if triplet_margin_loss(anchor=crop_d[anchor].unsqueeze(0),
                                           positive=catalog_d[p[0]].unsqueeze(0),
                                           negative=catalog_d[n].unsqueeze(0),
                                           margin=0.35) > 0:
                        triplets.append((anchor, p[0], n))              # p[0] --connected to x[1] below // p to x[1][0]
                    else:
                        num_zero += 1
                else:
                    n_in_pos += 1
                    print('n_in_pos value found! q, n: ', anchor, n)    # TODO: REMOVE THIS

        torch.save(triplets, 'triplet_cache.pkl')
        #torch.save(anc_to_cat_dist, "Anc_to_cat_cache.pkl")
        #torch.save(positive_only, "positive_only_cache.pkl")
        with open("hardtrain.csv", "w", newline='') as csvFile:  # base_dir
            writer = csv.writer(csvFile)
            triplets = [[self.path_to_crops + "/" + x[0] + ".jpg", self.path_to_catalog + "/" + x[1] + ".jpg",
                         self.path_to_catalog + "/" + x[2] + ".jpg"] for x in triplets]
            print("Process Completed\nNumber of triplets: ", len(triplets))
            writer.writerows(triplets)
            del triplets

        print('Number of zero losses: ', num_zero)
        print('Number of n in pos: ', n_in_pos)

        return
