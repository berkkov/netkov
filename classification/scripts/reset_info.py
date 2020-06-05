import torch


class ResetInfo(object):
    """
    Replaces certain all white pixels in the image with a given RGB values or a single integer for all three channels.
    """
    def __init__(self, fill_value=0):
        """

        :param fill_value:
        """
        self.fill_value = fill_value

    def __call__(self, img_tensor):
        """

        :param img_tensor:
        :return:
        """
        if self.fill_value == 0:
            black_shape = list(img_tensor[:, ((img_tensor[0, :, :] == 1) &
                                              (img_tensor[1, :, :] == 1) &
                                              (img_tensor[2, :, :] == 1))]
                               .shape)
            black_pixels = torch.zeros(black_shape)
            img_tensor[:,
                ((img_tensor[0, :, :] == 1) & (img_tensor[1, :, :] == 1) & (img_tensor[2, :, :] == 1))] = black_pixels
        else:
            black_shape = list(img_tensor[:, ((img_tensor[0, :, :] == 1) &
                                              (img_tensor[1, :, :] == 1) &
                                              (img_tensor[2, :, :] == 1))]
                               .shape)
            black_shape = [black_shape[1]]
            h_w_mask = ((img_tensor[0, :, :] == 1) & (img_tensor[1, :, :] == 1) & (img_tensor[2, :, :] == 1))
            img_tensor[0, h_w_mask] \
                = torch.full(black_shape, self.fill_value[0])

            img_tensor[1, h_w_mask] \
                = torch.full(black_shape, self.fill_value[1])

            img_tensor[2, h_w_mask] \
                = torch.full(black_shape, self.fill_value[2])
        return img_tensor


"""
if __name__ == '__main__':
    base_dir = 'C:\\Users\\user\\Downloads\\netkov_data\\all\\'
    dress_list = [x.split('\\')[-1] for x in glob.glob(f'{base_dir}\\*')]
    save_dir = 'C:\\Users\\user\\Desktop\\netkov_data\\all\\'

    count = 0
    for dress_id in dress_list:
        if not os.path.exists(save_dir + str(dress_id)):
            os.mkdir(save_dir + str(dress_id))

        img_list = glob.glob(base_dir + dress_id + '\\*.jpg') + glob.glob(base_dir + dress_id + '\*\*.jpg')
        cat_list = [dir for dir in glob.glob(base_dir + dress_id + '\\*') if 'jpg' not in dir]
        for cat_path in cat_list:
            if not os.path.exists(cat_path.replace('Downloads', 'Desktop')):
                os.mkdir(cat_path.replace('Downloads', 'Desktop'))
        for img_path in img_list:
            loader = LoadImage()
            img = loader(img_path)

            tensorized = transforms.ToTensor()(img)
            img_zeroed = reset_info(tensorized, fill_value=[0.485, 0.456, 0.406])

            img_zeroed = transforms.ToPILImage()(img_zeroed).convert("RGB")
            img_zeroed.save(img_path.replace('Downloads', 'Desktop'))
            count += 1
            print(f'\rProgress: {count}/ 12672')
"""