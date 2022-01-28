import argparse
import os
import torch

import torch.nn.functional as F

from torch import nn
from load_data import DataGenerator
from dnc import DNC
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter


class MANN(nn.Module):

    def __init__(self, num_classes, samples_per_class, model_size=128, input_size=784):
        super(MANN, self).__init__()

        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_size = input_size
        self.layer1 = nn.LSTM(num_classes + input_size,
                              model_size,
                              batch_first=True)
        self.layer2 = nn.LSTM(model_size,
                              num_classes,
                              batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

        # self.dnc = DNC(
        #     input_size=num_classes + input_size,
        #     output_size=num_classes,
        #     hidden_size=model_size,
        #     rnn_type='lstm',
        #     num_layers=1,
        #     num_hidden_layers=1,
        #     nr_cells=num_classes,
        #     cell_size=64,
        #     read_heads=1,
        #     batch_first=True,
        #     gpu_id=0,
        # )

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: tensor
                A tensor of shape [B, K+1, N, 784] of flattened images

            labels: tensor:
                A tensor of shape [B, K+1, N, N] of ground truth labels
        Returns:

            out: tensor
            A tensor of shape [B, K+1, N, N] of class predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # SOLUTION:
        examples = torch.cat([input_images, input_labels],
                             dim=-1)  # (B, K+1, N, 784 + N)
        B, R, N, D = examples.shape
        examples = examples.reshape(B, -1, D)

        support_examples = examples[:, :-N, :]  # (B, K * N, 784 + N)
        test_examples = examples[:, -N:, :]  # (B, N, 784 + N)
        real_test_labels = test_examples[:, :, 784:]  # (B, N, N)
        zero_labels = torch.zeros_like(real_test_labels)
        test_examples[:, :, 784:] = zero_labels

        # (B, (K+1) * N, 784 + N)
        examples = torch.cat([support_examples, test_examples], dim=1)
        outputs, _ = self.layer1(examples)  # (B, (K+1) * N, 128)
        outputs, _ = self.layer2(outputs)
        return outputs.reshape(B, R, N, N)

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: tensor
                A tensor of shape [B, K+1, N, N] of network outputs

            labels: tensor
                A tensor of shape [B, K+1, N, N] of class labels

        Returns:
            scalar loss
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # SOLUTION:
        test_preds = preds[:, -1, :, :]  # (B, N, N)
        test_labels = labels[:, -1, :, :]  # (B, N, N)
        test_labels = torch.argmax(test_labels, dim=-1)
        return F.cross_entropy(test_preds, test_labels)


def train_step(images, labels, model, optim):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def main(config):
    device = torch.device("cpu")
    writer = SummaryWriter(config.logdir)

    # Download Omniglot Dataset
    if not os.path.isdir('./omniglot_resized'):
        gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                            dest_path='./omniglot_resized.zip',
                                            unzip=True)
    assert os.path.isdir('./omniglot_resized')

    # Create Data Generator
    data_generator = DataGenerator(config.num_classes,
                                   config.num_samples,
                                   device=device)

    # Create model and optimizer
    model = MANN(config.num_classes, config.num_samples,
                 model_size=config.model_size)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(config.training_steps):
        images, labels = data_generator.sample_batch(
            'train', config.meta_batch_size)
        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch('test',
                                                         config.meta_batch_size)
            pred, test_loss = model_eval(images, labels, model)
            pred = torch.reshape(pred, [-1,
                                        config.num_samples + 1,
                                        config.num_classes,
                                        config.num_classes])
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            labels = torch.argmax(labels[:, -1, :, :], axis=2)

            writer.add_scalar('Train Loss', train_loss.cpu().numpy(), step)
            writer.add_scalar('Test Loss', test_loss.cpu().numpy(), step)
            writer.add_scalar('Meta-Test Accuracy',
                              pred.eq(labels).double().mean().item(),
                              step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--logdir', type=str,
                        default='run/log')
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--model_size', type=int, default=128)
    main(parser.parse_args())

    # Do not have GPU on me, so I can't do the DNC stuff unfortunately.
