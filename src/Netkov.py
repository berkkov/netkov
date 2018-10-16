import torch.nn as nn
import torch
import torch.nn.functional as F
from ShallowNet import ShallowNet
from InceptionNet import Inception
import NetworkUtils as util
from torchvision.models import resnet101


class ConnectedLayers(nn.Module):
    def __init__(self):
        super(ConnectedLayers, self).__init__()

        self.linear1 = nn.Linear(6336, 3072)
        self.linrelu1 = nn.ReLU(3072)
        self.dropout1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(3072, 2048)
        self.linrelu2 = nn.ReLU(2048)
        self.dropout2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(2048, 1024)       # TODO: Relu????

    def forward(self, x):
        x = self.linear1(x)
        x = self.linrelu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.linrelu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)

        x = F.normalize(x, p=2, dim=1)
        return x


class Netkov(nn.Module):
    def __init__(self):
        super(Netkov, self).__init__()

        self.ShallowNet = ShallowNet()
        self.backbone = Inception()
        self.linears = ConnectedLayers()

    def forward(self, input):
        x0 = self.backbone(input)  # Output of the CNN
        x1 = self.ShallowNet(input)  # Concatenated output of the two shallow nets

        x = torch.cat((x0, x1), 1)
        x = self.linears(x)
        return x


class TripletNetkov(nn.Module):
    def __init__(self):
        super(TripletNetkov, self).__init__()
        self.EmbeddingNet = Netkov().cuda()
        n_parameters = sum([p.data.nelement() for p in self.EmbeddingNet.parameters()])
        print('  + Number of params: {}'.format(n_parameters))


    def forward(self, query, positive, negative):
        q = self.EmbeddingNet(query)
        p = self.EmbeddingNet(positive)
        n = self.EmbeddingNet(negative)

        return q, p, n

    def get_embedding(self, x):
        return self.EmbeddingNet(x)


class SeperatedTripletNetkov(nn.Module):
    def __init__(self):
        super(SeperatedTripletNetkov, self).__init__()

        self.CatalogEmbedder = Netkov().cuda()
        self.QueryEmbedder = Netkov().cuda()

    def forward(self, query, positive, negative):
        query = self.QueryEmbedder(query)
        positive = self.CatalogEmbedder(positive)
        negative = self.CatalogEmbedder(negative)

        return query, positive, negative


class EvaNetkov(nn.Module):
    def __init__(self):
        super(EvaNetkov, self).__init__()
        self.EmbeddingNet = Netkov()
        n_parameters = sum([p.data.nelement() for p in self.EmbeddingNet.parameters()])
        print('  + Number of params: {}'.format(n_parameters))

    def forward(self, img):
        embedding = self.EmbeddingNet(img)

        return embedding

    def get_embedding(self, x):
        return self.EmbeddingNet(x)


class EvaNetkovCatalog(nn.Module):
    def __init__(self):
        super(EvaNetkovCatalog, self).__init__()
        self.CatalogEmbedder = Netkov().cuda()

    def forward(self, x):
        return self.CatalogEmbedder(x)


class EvaNetkovQuery(nn.Module):
    def __init__(self):
        super(EvaNetkovQuery, self).__init__()
        self.QueryEmbedder = Netkov().cuda()

    def forward(self, x):
        return self.QueryEmbedder(x)


def pass_triplet(model, triplet, loader):
    """
    Pass triplet through a model
    :param model:
    :param triplet: in shape [q_path, p_path, n_path]
    :return: fv_q, fv_p, fv_n
    """
    input_triplet = loader.load_batch(triplet).cuda()
    out = model(input_triplet)
    fv_q = out[0]
    fv_p = out[1]
    fv_n = out[2]

    return fv_q, fv_p, fv_n


def Netkov_batch_eval(weight_path, image_paths, batch_size=88):
    model = Netkov().cuda()
    std = torch.load(weight_path)['state_dict']
    model.load_state_dict(std)      #TODO: Path
    model.eval()
    fv_dict = {}

    data_loader = torch.utils.data.DataLoader(util.FVBatchLoader(image_paths), batch_size=batch_size, shuffle=False,
                                              num_workers=0)
    count = 0

    with torch.no_grad():
        for id, batch in data_loader:
            batch = batch.cuda()
            out = model(batch)
            count += 1
            del batch
            model.zero_grad()
            print(str(count) + "/" + str(len(data_loader)))
            for i in range(0, len(id)):
                fv_dict[id[i]] = out[i]

            torch.cuda.empty_cache()

    print(len(fv_dict))
    torch.save(fv_dict, "D:\\clean26_crop_fv.pth")
    return
