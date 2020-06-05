import torch.nn as nn
import pretrainedmodels
import torch


def _resnet50():
    basemodel = pretrainedmodels.__dict__["resnet50"](num_classes=1000)

    model = nn.Sequential(
        basemodel.conv1,
        basemodel.bn1,
        basemodel.relu,
        basemodel.maxpool,

        basemodel.layer1,
        basemodel.layer2,
        basemodel.layer3,
        basemodel.layer4
    )

    input_config = {'input_space': basemodel.input_space,
                    'input_range': basemodel.input_range,
                    'input_size': basemodel.input_size,
                    'std': basemodel.std,
                    'mean': basemodel.mean}
    return model, input_config


class NetkovSOTA(nn.Module):
    """

    """
    def __init__(self, embedding_size):
        super(NetkovSOTA, self).__init__()
        # Backbone: Resnet50 outputs (N, 2048, 7, 7) tensor.
        self.backbone, self.input_config = _resnet50()

        # Average pool each filter's 7x7 frame to get (N, 2048, 1, 1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.layer_norm = nn.LayerNorm(normalized_shape=512*4,
                                       elementwise_affine=False)

        self.fc1 = nn.Linear(in_features=512*4,
                             out_features=embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.layer_norm(x)
        x = self.fc1(x)
        # L2 Normalization of feature embeddings
        x = nn.functional.normalize(x, p=2)
        return x


if __name__ == '__main__':
    from src.NetworkUtils import SingleImageLoader
    loader = SingleImageLoader(network_type='resnet')
    img = loader.load_image('C:\\Users\\user\\Downloads\\netkov_data\\all\\1\\cat\\340000.jpg')
    img.shape
    avg_pooling = nn.AdaptiveAvgPool2d(1)
    nn.LayerNorm(input_dim, elementwise_affine=False)
    avg_pooling(img).view(avg_pooling(img).size(0), -1)

    model = NetkovSOTA(512)
    type(img.float())
    model(img.type(torch.FloatTensor)).shape
