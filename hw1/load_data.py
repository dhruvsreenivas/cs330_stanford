import numpy as np
import os
import random
import torch
from sklearn.utils import shuffle


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        def sampler(x): return random.sample(x, nb_samples)
    else:
        def sampler(x): return x
    # print(type(paths[0]))
    # imgs = sampler(os.listdir(paths[0]))
    # print(imgs)
    images_labels = [(i, os.path.join(path, image.decode()))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}, device=torch.device('cpu')):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)

            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.

            device: cuda.device: 
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]
        self.device = device

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from

            batch_size: int:
                Size of batch of tasks to sample

        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class, 
                N is number of classes

            labels: tensor
                A tensor of images of size [B, K+1, N, N] 
                where B is batch size, K is number of samples per class, 
                N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # SOLUTION:
        img_batch = []
        label_batch = []
        for _ in range(batch_size):
            sampled_classes = np.random.choice(
                folders, self.num_classes, replace=False)

            labels = np.eye(self.num_classes)  # (N, N)

            lbls_imgs = get_images(
                sampled_classes, labels, nb_samples=self.num_samples_per_class + 1, shuffle=False)

            support_imgs, support_labels, query_imgs, query_labels = [], [], [], []
            for idx, (label, img) in enumerate(lbls_imgs):
                if idx % (self.num_samples_per_class + 1) == 0:
                    query_imgs.append(image_file_to_array(img, self.dim_input))
                    query_labels.append(label)
                else:
                    support_imgs.append(
                        image_file_to_array(img, self.dim_input))
                    support_labels.append(label)

            support_imgs = np.stack(
                support_imgs, axis=0).reshape(-1, self.num_classes, self.dim_input)
            support_labels = np.stack(
                support_labels).reshape(-1, self.num_classes, self.num_classes)
            query_imgs = np.stack(query_imgs, axis=0).reshape(
                1, self.num_classes, self.dim_input)
            query_labels = np.stack(
                query_labels).reshape(-1, self.num_classes, self.num_classes)

            # shuffle image/label pairs
            support_imgs, support_labels = shuffle(
                support_imgs, support_labels)
            query_imgs, query_labels = shuffle(query_imgs, query_labels)

            images = np.concatenate([support_imgs, query_imgs], axis=0)
            labels = np.concatenate([support_labels, query_labels], axis=0)

            img_batch.append(images)
            label_batch.append(labels)

        img_batch = np.stack(img_batch, axis=0)
        label_batch = np.stack(label_batch, axis=0)
        # print(img_batch.shape)
        # print(label_batch.shape)
        return torch.from_numpy(img_batch).float(), torch.from_numpy(label_batch).float()


if __name__ == '__main__':
    dg = DataGenerator(5, 2)
    img_batch, lbl_batch = dg.sample_batch('train', 128)
    print(img_batch.shape, lbl_batch.shape)
