import torch.nn as nn
import torch
import torch.nn.functional as F
from ShallowNet import ShallowNet
from InceptionNet import Inception
import NetworkUtils as util
from Resnet import resnet34, resnet50, resnet101
from torchvision.models import resnet101


class ConnectedLayers(nn.Module):
    """
        Linear layers after power model and shallow nets
    """
    def __init__(self, backbone, include_shallow_net=True):
        super(ConnectedLayers, self).__init__()

        self.backbone_name = backbone
        output_size = {'resnet34': 512, 'resnet50': 512*4, 'resnet101': 512*4, 'inception': 1536}

        first_lin = output_size[self.backbone_name]
        if include_shallow_net:
            first_lin += 4800

        self.linear1 = nn.Linear(first_lin, 3072)
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
    def __init__(self, include_shallow_net=True, backbone='inception'):
        """
            Initialize the power network with different settings.
            :param backbone: Name of the power model
            'inception' (v4) or 'resnet34', 'resnet50' 'resnet101' models are supported.
            :param include_shallow_net: If True, 2 side Shallow Networks will be added to backbone.
            If false, it basically returns the backbone model - last layer + couple of linear layers.
        """
        assert backbone in ['inception', 'resnet34', 'resnet50', 'resnet101']
        super(Netkov, self).__init__()

        self.backbone_name = backbone
        self.include_shallow_net = include_shallow_net

        if self.include_shallow_net:
            self.ShallowNet = ShallowNet()

        if self.backbone_name == 'inception':
            self.backbone = Inception()
            print('***', self.backbone_name, '*** created')
        elif self.backbone_name == 'resnet34':
            self.backbone = resnet34()
            print('***', self.backbone_name, '*** created')
        elif self.backbone_name =='resnet50':
            self.backbone = resnet50()
            print('***', self.backbone_name, '*** created')
        elif self.backbone_name == 'resnet101':
            self.backbone = resnet101()
            print('***', self.backbone_name, '*** created')

        self.linears = ConnectedLayers(backbone=self.backbone_name, include_shallow_net=self.include_shallow_net)

    def forward(self, input):
        if self.include_shallow_net:
            x0 = self.backbone(input)  # Output of the CNN
            x1 = self.ShallowNet(input)  # Concatenated output of the two shallow nets

            x = torch.cat((x0, x1), 1)
            x = self.linears(x)
        else:
            x = self.backbone(input)
            x = self.linears(x)
        return x


class TripletNetkov(nn.Module):
    """
        This is triplet network class for using 1 network for both query and catalog images.
    """
    def __init__(self, backbone, include_shallow_net):
        """
        Initialize the triplet net for with different settings.
        :param backbone: Name of the power model for the triplet network.
        'inception' (v4) or 'resnet34', 'resnet50' 'resnet101' models are supported.
        :param include_shallow_net: If True, 2 side Shallow Networks will be added to backbone.
        """
        super(TripletNetkov, self).__init__()

        self.include_shallow_net = include_shallow_net
        self.backbone_name = backbone
        self.EmbeddingNet = Netkov(backbone=backbone, include_shallow_net=self.include_shallow_net).cuda()
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
    def __init__(self, backbone, include_shallow_net):
        super(SeperatedTripletNetkov, self).__init__()

        self.include_shallow_net = include_shallow_net
        self.backbone_name = backbone
        self.CatalogEmbedder = Netkov(backbone=backbone, include_shallow_net=self.include_shallow_net).cuda()
        self.QueryEmbedder = Netkov(backbone=backbone, include_shallow_net=self.include_shallow_net).cuda()

    def forward(self, query, positive, negative):
        query = self.QueryEmbedder(query)
        positive = self.CatalogEmbedder(positive)
        negative = self.CatalogEmbedder(negative)

        return query, positive, negative


class EvaNetkov(nn.Module):
    def __init__(self, backbone, include_shallow_net):
        super(EvaNetkov, self).__init__()
        self.EmbeddingNet = Netkov(backbone=backbone, include_shallow_net=include_shallow_net)
        n_parameters = sum([p.data.nelement() for p in self.EmbeddingNet.parameters()])
        print('  + Number of params: {}'.format(n_parameters))

    def forward(self, img):
        embedding = self.EmbeddingNet(img)

        return embedding

    def get_embedding(self, x):
        return self.EmbeddingNet(x)


class EvaNetkovCatalog(nn.Module):
    def __init__(self, backbone, include_shallow_net):
        super(EvaNetkovCatalog, self).__init__()
        self.CatalogEmbedder = Netkov(backbone=backbone, include_shallow_net=include_shallow_net).cuda()

    def forward(self, x):
        return self.CatalogEmbedder(x)


class EvaNetkovQuery(nn.Module):
    def __init__(self, backbone, include_shallow_net):
        super(EvaNetkovQuery, self).__init__()
        self.QueryEmbedder = Netkov(backbone=backbone, include_shallow_net=include_shallow_net).cuda()

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
