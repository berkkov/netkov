import torch.utils.data
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import math
from src.NetworkUtils import ToRange255, ToSpaceBGR
import torch.nn as nn
from classification.scripts.reset_info import ResetInfo
import numpy as np




class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


class TransformImage(object):

    def __init__(self, input_config, pad_value=None, random_crop=False, random_hflip=False, normalize=True):

        self.input_size = input_config['input_size']
        self.input_space = input_config['input_space']
        self.input_range = input_config['input_range']
        self.mean = input_config['mean']
        self.std = input_config['std']
        self.pad_value = pad_value if pad_value else self.mean
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        tfs = []

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            # tfs.append(transforms.CenterCrop(max(self.input_size))) # TODO:Maybe use center crops?
            pass
        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        # tfs.append(transforms.ToTensor())

        resizer = PaddedResize(max(self.input_size), pad_value=self.pad_value)
        tfs.append(resizer)
        pxl_replacer = ResetInfo(fill_value=self.mean)
        tfs.append(pxl_replacer)

        tfs.append(ToSpaceBGR(self.input_space == 'BGR'))
        tfs.append(ToRange255(max(self.input_range) == 255))
        if normalize:
            tfs.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class PaddedResize(object):
    """
    Resize images by preserving the aspect ratio and zero or constant padding around the image. Uses larger dimension
    of images to downsample. Works only for square image outputting. Also converts the image to tensor by default.
    """
    def __init__(self, out_dim, pad_value=0, return_img=False):
        assert isinstance(out_dim, int)
        self.out_dim = out_dim
        self.return_img = return_img
        self.pad_value = pad_value

    def __call__(self, img):
        """

        :param img:
        :return:
        """
        # Find the maximum dimension of the image and number of pixels in that direction
        img_max_dim = max(img.size)
        height = int(img.size[1] / img_max_dim * self.out_dim)
        width = int(math.ceil(img.size[0] / img_max_dim * self.out_dim))

        # Calculate the required padding amount and dimension of padding
        pad_amount = max(self.out_dim - width, self.out_dim - height)
        pad_loc = np.argmax([self.out_dim - width, self.out_dim - height])
        padding = [0, 0, 0, 0]

        # 0 left, 1 right, 2 top, 3 bottom
        if pad_loc == 0:
            if pad_amount % 2 == 0:
                padding[0] = pad_amount / 2
                padding[1] = pad_amount / 2
            else:
                padding[0] = (pad_amount + 1) / 2
                padding[1] = (pad_amount - 1) / 2
        elif pad_loc == 1:
            if pad_amount % 2 == 0:
                padding[2] = pad_amount / 2
                padding[3] = pad_amount / 2
            else:
                padding[2] = (pad_amount + 1) / 2
                padding[3] = (pad_amount - 1) / 2

        padding = [int(pad) for pad in padding]

        # Resize image
        test = transforms.Resize((height, width))(img)

        # Convert image to tensor and apply zero padding
        tensorized = transforms.ToTensor()(test)

        if self.pad_value == 0:
            padded_tensor = nn.ZeroPad2d(tuple(padding))(tensorized)
        else:
            channel1_pad = nn.ConstantPad2d(tuple(padding), self.pad_value[0])(tensorized[0])
            channel2_pad = nn.ConstantPad2d(tuple(padding), self.pad_value[1])(tensorized[1])
            channel3_pad = nn.ConstantPad2d(tuple(padding), self.pad_value[2])(tensorized[2])
            padded_tensor = torch.stack([channel1_pad, channel2_pad, channel3_pad])

        if self.return_img:
            return transforms.ToPILImage()(padded_tensor).convert("RGB")
        else:
            return padded_tensor


class TrainLoader(torch.utils.data.Dataset):
    def __init__(self, sampled_img_to_proxy_path, input_config, path_prefix):
        self.loader = LoadImage()
        self.transformer = TransformImage(input_config, random_hflip=True)
        self.img_to_label = torch.load(sampled_img_to_proxy_path)
        self.img_path_list = list(self.img_to_label.keys())
        self.path_prefix = path_prefix

    def __getitem__(self, index):
        x = torch.autograd.Variable(self.transformer(self.loader(self.path_prefix + self.img_path_list[index])))
        label = self.img_to_label[self.img_path_list[index]]

        return x, label

    def __len__(self):
        return len(self.img_path_list)


class TrainSampler(object):
    def __init__(self, img_to_proxy_path, batch_size, num_samples_per_class=5, replacement=True,
                 replacement_labels=True):
        #super(TrainSampler, self).__init__()
        self.num_samples_per_class = num_samples_per_class
        self.img_to_label = torch.load(img_to_proxy_path)
        self.label_to_img = self._get_proxy_to_img()
        self.labels = list(self.label_to_img.keys())
        self.batch_size = batch_size
        self.replacement = replacement
        self.replacement_labels = replacement_labels
        self.previously_sampled = []
        self.img_path_list = list(self.img_to_label.keys())

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample_batch()
        self.previously_sampled = []

    def sample_batch(self):
        num_classes = self.batch_size // self.num_samples_per_class
        if self.replacement_labels:
            sampled_labels = np.random.choice(self.labels, num_classes, replace=False)
        else:
            sampled_labels = np.random.choice([lbl for lbl in self.labels if lbl not in self.previously_sampled],
                                              num_classes,
                                              replace=False)
            self.previously_sampled.extend(sampled_labels)

        sampled_imgs = []
        for label in sampled_labels:
            sampled_imgs.extend(np.random.choice(self.label_to_img[label],
                                                 self.num_samples_per_class,
                                                 replace=self.replacement))
        sampled_indices = [self.img_path_list.index(ipath) for ipath in sampled_imgs]
        return sampled_indices

    def _get_proxy_to_img(self):
        label_to_img = {}
        for img_path in self.img_to_label.keys():
            label = self.img_to_label[img_path]
            if label not in label_to_img:
                label_to_img[label] = []
                label_to_img[label].append(img_path)
            else:
                label_to_img[label].append(img_path)
        return label_to_img

    def __len__(self):
        return len(self.labels) // self.batch_size


class TestLoader(torch.utils.data.Dataset):
    def __init__(self, img_to_proxy_path, input_config, path_prefix, catalog_crowders_to_proxy_path=None,
                 crowder_size=10000):
        self.loader = LoadImage()
        self.transformer = TransformImage(input_config)
        self.img_to_label = torch.load(img_to_proxy_path)
        self.img_path_list = list(self.img_to_label.keys())
        self.path_prefix = path_prefix
        if catalog_crowders_to_proxy_path:
            self.catalog_crowders_to_label = torch.load(catalog_crowders_to_proxy_path)
            self.img_path_list.extend(list(self.catalog_crowders_to_label.keys())[:crowder_size])

    def __getitem__(self, index):
        x = torch.autograd.Variable(self.transformer(self.loader(self.path_prefix + self.img_path_list[index])))
        img_path = self.img_path_list[index]

        return x, img_path

    def __len__(self):
        return len(self.img_path_list)


if __name__ == '__main__':
    loader = LoadImage()
    img = loader('C:\\Users\\user\\Downloads\\netkov_data\\all\\21\\349156.jpg')
    img2 = loader('C:\\Users\\user\\Downloads\\netkov_data_2\\10090\\cat\\60819.jpg')
    transformer = TransformImage(model.input_config, pad_value=0)
    img = transformer(img)
    img2 = transformer(img2)
    imgs = torch.stack([img, img2])
    imgs.shape
    make_grid(imgs).shape
    transforms.ToPILImage()(make_grid(imgs)).convert("RGB").show()
    """
    transforms.ToTensor()(img)
    seed = 23
    np.random.seed(seed)
    transformer = TransformImage(model.input_config, pad_value=0)
    takeit = transformer(img)
    transforms.ToPILImage()(takeit).convert("RGB").show()
    takeit
    # im.save('C:\\Users\\user\\Desktop\\img_tensorized.jpg')"""



class Recommend(object):
    def __init__(self, query_img_name, ranked_indices, query_paths, catalog_paths, input_config,
                 n_recommendations=10, ignore_first=True):
        """

        :param query_img_name:
        :param ranked_indices:
        :param query_fvs:
        :param catalog_fvs:
        :param query_paths:
        :param catalog_paths:
        :param n_recommendations:
        """
        self.query_img_name = query_img_name # +
        self.query_index = query_paths.index(query_img_name) # +
        self.sorted_indices = ranked_indices
        self.query_paths = query_paths
        self.catalog_paths = catalog_paths
        self.n_recommendations = n_recommendations
        self.first_rec_index = 1 if ignore_first else 0

        self.loader = LoadImage()
        self.converter = TransformImage(input_config=input_config, normalize=False)

    def _get_top_n_indices(self):
        top_n_indices = self.sorted_indices[self.query_index][self.first_rec_index:
                                                              self.first_rec_index+self.n_recommendations]
        grid = [self.converter(self.loader(self.query_paths[self.query_img_name]))]
        for cat_idx in top_n_indices:
            grid.append(self.converter(self.loader(self.catalog_paths[cat_idx])))

        grid_tensor = torch.stack(grid)
        grid_image = make_grid(grid_tensor)
        grid_image = transforms.ToPILImage()(grid_image).convert("RGB")
        grid_image.show()
        return grid_image
