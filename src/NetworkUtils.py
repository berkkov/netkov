from PIL import Image
from torchvision import transforms
import math
import torch.nn as nn
import torch
import csv
import torch.utils.data
import torch.utils.data.sampler
import copy
import numpy as np
import os
import matplotlib.pyplot as plt


class Visualizer(object):
    def __init__(self, model, weights_path):
        self.model = model
        self.weights_path = weights_path
        self.model.load_state_dict(torch.load(weights_path), strict=False)
        self.summary = {}

    def get_activations(self, input, layer_no):
        cropped_model = CroppedNet(self.model, layer_no)
        cropped_model.eval()
        activations = cropped_model(input)

        return activations

    def visualize_activations(self, activations, layer_no, save_dir="D:\\netkov_diagnostics\\activations"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_filters = activations.shape[1]
        for filter in range(num_filters):
            act = activations[:, filter, :, :].detach().cpu()
            act = act.transpose(0, 2).numpy().astype(np.uint8)
            act = act[:, :, 0]
            act = act * 255
            img = Image.fromarray(act, "L")
            img.save(os.path.join(save_dir, str(layer_no) + '_' + str(filter) + '.jpg'))

        print("Activation visuals are saved to ", save_dir)

    def log_activation_stats(self, activations, layer_no, save_dir="D:\\netkov_diagnostics\\activation_stats"):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        num_filters = activations.shape[1]
        for filter in range(num_filters):
            filter_key = str(layer_no) + '_' + str(filter)
            act = activations[0, filter, :, :]
            act_np = act.detach().cpu().numpy()

            hist, bins = np.histogram(act_np, bins=np.linspace(0, 2, num=210))
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)
            plt.savefig(os.path.join(save_dir, filter_key + ".png"))
            plt.close()
            mean = act.mean().item()

            summary = {'mean': mean}
            self.summary[filter_key] = summary

    def log_weights(self, layer_no, save_dir="D:\\netkov_diagnostics\\weight_stats"):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        cropped_model = CroppedNet(self.model, layer_no)
        std = cropped_model.state_dict()
        for layer_name in std:
            if 'groupnorm' not in layer_name and 'linear' not in layer_name:
                print(layer_name)
                layer_weights = std[layer_name]

                for filter_no in range(layer_weights.size()[0]):                # TODO: Implement a macro version
                    filter_key = layer_name + "__" + str(filter_no)
                    weights = layer_weights[filter_no].cpu()
                    mean = weights.mean().item()

                    weights = weights.cpu().numpy()
                    standard_dev = np.std(weights)
                    percentiles = np.percentile(weights, q=np.linspace(0, 100, 5))  # TODO: Graph

                    if mean == 0 and standard_dev == 0:
                        print("Mean and STD of weights at this kernel is 0. Analyze ", filter_key)

                    hist, bins = np.histogram(weights, bins=np.linspace(0, 0.02, num=21))
                    width = 0.7 * (bins[1] - bins[0])
                    center = (bins[:-1] + bins[1:]) / 2
                    plt.bar(center, hist, align='center', width=width)
                    plt.savefig(os.path.join(save_dir, filter_key + ".png"))
                    plt.close()


class CroppedNet(nn.Module):
    def __init__(self, model, layer_num):
        super(CroppedNet,self).__init__()
        if layer_num == 'all':
            self.cropped_net = self.cropped_net = nn.Sequential(*list(model.QueryEmbedder.backbone.features.children())[:])
            print(list(model.QueryEmbedder.backbone.features.children())[:])
        else:
            self.cropped_net = nn.Sequential(*list(model.QueryEmbedder.backbone.features.children())[:layer_num+1])
            print(list(model.QueryEmbedder.backbone.features.children())[layer_num])

    def forward(self, input):
        return self.cropped_net(input)


class TripletLoader(torch.utils.data.Dataset):
    def __init__(self, triplets_path):
        with open(triplets_path) as file:
            reader = csv.reader(file)
            self.triplets = [triplet for triplet in reader]
        self.loader = LoadImage()
        self.transformer = TransformImage()

    def __getitem__(self, index):

        q = torch.autograd.Variable(self.transformer(self.loader(self.triplets[index][0])))     #.unsqueeze(0)
        p = torch.autograd.Variable(self.transformer(self.loader(self.triplets[index][1])))     #.unsqueeze(0)
        n = torch.autograd.Variable(self.transformer(self.loader(self.triplets[index][2])))     #.unsqueeze(0)

        return q, p, n

    def __len__(self):
        return len(self.triplets)


class ContinuedSampler(torch.utils.data.sampler.Sampler):
    """
    Custom sampler that uses the latest saved shuffler. Required for resuming the model that was last saved before an
    epoch ends.
    """
    def __init__(self, data_source, sequence_path, last_batch_id, batch_size):
        self.data_source = data_source
        self.sequence = list(torch.load(sequence_path).values())
        self.start_batch_id = last_batch_id
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.sequence[(self.start_batch_id+1)*self.batch_size:])

    def __len__(self):
        return len(self.data_source)


class FVBatchLoader(torch.utils.data.Dataset):
    def __init__(self, imgs_path):

        self.images = [img for img in imgs_path]            # wtf
        self.loader = LoadImage()
        self.transformer = TransformImage()

    def __getitem__(self, index):           # TODO: First delimiter might bug between linux - windows
        return self.images[index].split('/')[-1].split(".")[0], torch.autograd.Variable(self.transformer(self.loader(self.images[index])))

    def __len__(self):
        return len(self.images)


class ToSpaceBGR(object):
    """
    Converts BGR to RGB
    """
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 preserve_aspect_ratio=False):

        self.input_size = [3, 299, 299]
        self.input_space = 'RGB'
        self.input_range = [0, 1]
        self.mean = [0.5, 0.5, 0.5] # for Imagenet
        self.std = [0.5, 0.5, 0.5]  # for Imagenet
        #self.mean = [0.6544, 0.6116, 0.6016]
        #self.std = [0.3459, 0.3541, 0.3487]

        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size) / self.scale))))
        else:
            height = int(self.input_size[1]) # / self.scale)
            width = int(self.input_size[2]) # / self.scale)
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            #tfs.append(transforms.CenterCrop(max(self.input_size)))        #TODO:Maybe use center crops?
            pass
        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space == 'BGR'))
        tfs.append(ToRange255(max(self.input_range) == 255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


class BatchLoader(object):

    def __init__(self, batch_size):

        self.batch_size = batch_size
        self.img_loader = LoadImage()
        self.transformer = TransformImage()

    def load_batch(self, batch):
        '''

        :param batch: list of paths
        :return:
        '''
        out = torch.Tensor()
        count = 0

        for path in batch:
            img = self.img_loader(path)
            img = self.transformer(img)
            img = torch.autograd.Variable(img)
            img = img.unsqueeze(0)
            count += 1
            if count > 1:
                out = torch.cat((out, img), 0)
            else:
                out = img
        return out

    def get_ids(self, batch):
        ids = []
        for path in batch:
            id = path.split('\\')[-1].split(".")[0]
            ids.append(id)

        return ids


class SingleImageLoader(object):
    def __init__(self):
        self.img_loader = LoadImage()
        self.transformer = TransformImage()
    def load_image(self,img_path):
        image = self.img_loader(img_path)
        image = self.transformer(image)
        image = torch.autograd.Variable(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        return image


class TripletDataset(object):
    def __init__(self, triplets_path):
        """

        :param triplets_file: List of lists containing all the triplets and corresponding image paths.
        e.g [[q1_path, p1_path, n1_path], [q2_path, p2_path, n2_path] ...]
        """

        with open(triplets_path) as file:
            reader = csv.reader(file)
            self.triplets = [triplet for triplet in reader]
        self.loader = LoadImage()
        self.transformer = TransformImage()


    def load_single_triplet(self, index):
        """
        Returns processed tensors of triplet members q, p, n
        :param index: index of the triplet to be loaded
        :return: processed images ready to be fed into network
        """
        q = torch.autograd.Variable(self.transformer(self.loader(self.triplets[index][0]))).unsqueeze(0)
        p = torch.autograd.Variable(self.transformer(self.loader(self.triplets[index][1]))).unsqueeze(0)
        n = torch.autograd.Variable(self.transformer(self.loader(self.triplets[index][2]))).unsqueeze(0)

        return q, p, n

    def load_batch_triplet(self, batch_size, batch_num):
        """
        Returns specific batch of triplets
        :param batch_size: Batch size (Number of triplets in a batch)
        :param batch_num: Batch to be loaded (e.g. 1 ...)
        :return:
        """

        img_count = 0
        self.batch = self.triplets[batch_size * (batch_num - 1) : batch_size * batch_num]
        for trip in self.batch:
            for img_path in trip:
                img = torch.autograd.Variable(self.transformer(self.loader(img_path))).unsqueeze(0)

        for triplet in self.batch:
            if not self.qs:
                self.qs = triplet[0]
                self.ps = triplet[1]
                self.ns = triplet[2]
            else:
                self.qs = torch.cat((self.qs, triplet[0]),0)
                self.ps = torch.cat((self.ps, self.batch[1]),0)
                self.ns = torch.cat((self.ns, self.batch[2]), 0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


def input_creator(image_path):
    image_loader = LoadImage()
    image_transformer = TransformImage()
    image_tensor = torch.autograd.Variable(image_transformer(image_loader(image_path)))
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def load_Inception_weights(model, weight_pth='weights/InceptionV4.pth'):
    state_dict = torch.load(weight_pth)
    del state_dict["last_linear.bias"]
    del state_dict["last_linear.weight"]
    model.load_state_dict(state_dict)
    return model


def fv_structurer(loader, batch, feature_vectors):
    """
    Structures and saves Network outputs
    :param feature_vectors: Network outputs
    :return:
    """
    ids = loader.get_ids(batch)
    fv_dict = {}
    for i in range(len(ids)):
        fv_dict[ids[i]] = feature_vectors[i]

    torch.save(fv_dict, "test")

    return fv_dict


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


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.5, -0.5, -0.5]
    reverse_std = [1/0.5, 1/0.5, 1/0.5]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


if __name__ == '__main__':
    import Netkov
    loader = SingleImageLoader()
    image = loader.load_image("D:\\test2.jpg")
    model = Netkov.EvaNetkovQuery()
    #for layer_no in range(0, 5):
        #print('*****' * 20)
        #print("Layer No: ", layer_no)
        #visualizer = Visualizer(model, 'D:\\Indirilenler\\adadelta_checkpoint_50_10.pth.tar')       #TODO: Is this necessary?
        #activations = visualizer.get_activations(image, layer_no=layer_no)
        #visualizer.visualize_activations(activations, layer_no)
        #visualizer.log_activation_stats(activations, layer_no)

    visualizer = Visualizer(model, 'D:\\Indirilenler\\adadelta_checkpoint_1450_19.pth.tar')
    visualizer.log_weights('all')
