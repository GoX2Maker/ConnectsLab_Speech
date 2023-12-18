import os

import torch

import wandb


def to_np(x):
    return x.cpu().numpy()


class TensorBoardLogger(object):
    def __init__(self, id, log_dir, log_params):
        os.makedirs(log_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)
        self.log_params = log_params

    def update(self, epoch, results_state, parameters=None):
        loss = results_state.loss_results[epoch]
        wer = results_state.wer_results[epoch]
        cer = results_state.cer_results[epoch]
        values = {
            'Avg Train Loss': loss,
            'Avg WER': wer,
            'Avg CER': cer
        }
        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)
        if self.log_params:
            for tag, value in parameters():
                tag = tag.replace('.', '/')
                self.tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                self.tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)

    def load_previous_values(self, start_epoch, result_state):
        loss_results = result_state.loss_results[:start_epoch]
        wer_results = result_state.wer_results[:start_epoch]
        cer_results = result_state.cer_results[:start_epoch]

        for i in range(start_epoch):
            values = {
                'Avg Train Loss': loss_results[i],
                'Avg WER': wer_results[i],
                'Avg CER': cer_results[i]
            }
            self.tensorboard_writer.add_scalars(self.id, values, i + 1)

class WandBLogger(object):
    def __init__(self, projectTitle, note, cfg) -> None:
        wandb.login()

        run = wandb.init(
        project=projectTitle,
        notes=note,
        config = cfg
        )

    def update(self, loss):
        values = {
            'train/Loss_val': loss.val,
            'train/Loss_avg':  loss.avg,
        }
        wandb.log({**values})

    def update_valid(self, epoch, results_state):
        loss = results_state.loss_results[epoch]
        wer = results_state.wer_results[epoch]
        cer = results_state.cer_results[epoch]
        values = {
            'valid/Avg Train Loss': loss,
            'valid/Avg WER': wer,
            'valid/Avg CER': cer
        }
        wandb.log({**values})

        


        