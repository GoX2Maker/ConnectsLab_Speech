#!/usr/bin/env python
# coding: utf-8


import argparse
import yaml
from omegaconf import DictConfig
import ast

import math
import json
import time
import random
from collections import OrderedDict
from tqdm import tqdm

from hydra.utils import to_absolute_path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, DSElasticDistributedSampler, AudioDataLoader
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.logger import WandBLogger
from deepspeech_pytorch.model import DeepSpeech, supported_rnns
from deepspeech_pytorch.state import TrainingState
from deepspeech_pytorch.utils import check_loss, CheckpointHandler
from deepspeech_pytorch.testing import evaluate


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        cfg = DictConfig(yaml.safe_load(file))
        cfg.optimizer.betas = ast.literal_eval(cfg.optimizer.betas)
        cfg.optimizer.eps = float(cfg.optimizer.eps)
        cfg.optimizer.weight_decay = float(cfg.optimizer.weight_decay)
        return cfg

def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepSpeech training script')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration yaml file')
    # parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides yaml config if provided)')
    # Add other command-line arguments as needed
    return parser.parse_args()

def getCERandWER(decoder, out, output_sizes, targets, target_sizes):
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    decoded_output, _ = decoder.decode(out, output_sizes)

    split_targets = []
    offset = 0
    for size in target_sizes:
        split_targets.append(targets[offset:offset + size])
        offset += size

    target_strings = decoder.convert_to_strings(split_targets)

    if DEBUG:
        print('target:',  target_strings[0][0])
        print('predicted:',decoded_output[0][0])

    for x in range(len(target_strings)):
        transcript, reference = decoded_output[x][0], target_strings[x][0]
        wer_inst = decoder.wer(transcript, reference)
        cer_inst = decoder.cer(transcript, reference)
        total_wer += wer_inst
        total_cer += cer_inst
        num_tokens += len(reference.split())
        num_chars += len(reference.replace(' ', ''))

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    return wer * 100, cer * 100, target_strings[0][0], decoded_output[0][0]




def train(cfg):
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)



    print(f'CUDA is available : {torch.cuda.is_available()}')
    print(f'CUDA Device Count  : {torch.cuda.device_count()}')
    device = torch.device(f"cuda:{cfg.training.cudaNumber}" if cfg.training.useCuda else "cpu")
    torch.cuda.set_device(cfg.training.cudaNumber)
    print(f'Selected CUDA Number: {torch.cuda.current_device()}')



    checkpoint_handler = CheckpointHandler(
                                        save_folder              = cfg.checkpointing.save_folder,
                                        best_val_model_name      = cfg.checkpointing.best_val_model_name,
                                        checkpoint_per_iteration = cfg.checkpointing.checkpoint_per_iteration,
                                        save_n_recent_models     = cfg.checkpointing.save_n_recent_models
                                        )

    if cfg.checkpointing.load_auto_checkpoint:
            latest_checkpoint = checkpoint_handler.find_latest_checkpoint()
            if latest_checkpoint:
                cfg.checkpointing.continue_from = latest_checkpoint

    if cfg.checkpointing.continue_from:  # Starting from previous model
            state = TrainingState.load_state(state_path=to_absolute_path(cfg.checkpointing.continue_from))
            model = state.model
            if cfg.training.finetune:
                state.init_finetune_states(cfg.training.epochs)
    else:
        with open(cfg.data.labels_path) as label_file:
            labels = json.load(label_file)


        audio_conf = dict(sample_rate     = cfg.data.sample_rate,
                            window_size   = cfg.data.window_size,
                            window_stride = cfg.data.window_stride,
                            window        = cfg.data.window)


        model = DeepSpeech(rnn_hidden_size    = cfg.model.hidden_size,
                                nb_layers     = cfg.model.hidden_layers,
                                bidirectional = cfg.model.bidirectional,
                                rnn_type      = supported_rnns[cfg.model.rnn_type],

                                labels        = labels,
                                audio_conf    = audio_conf,
                                )

        # rnn 타입 확인
        rnn_type = cfg.model.rnn_type
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"


        state = TrainingState(model=model)
        state.init_results_tracking(epochs=cfg.training.epochs)




    # Data setup
    evaluation_decoder = GreedyDecoder(model.labels)

    # Data path 정리 
    train_dataset = SpectrogramDataset(audio_conf           = model.audio_conf,
                                    manifest_filepath    = cfg.data.train_manifest,
                                    labels               = model.labels,
                                    normalize            = True,
                                    speed_volume_perturb = cfg.augmentation.speed_volume_perturb,
                                    spec_augment         = cfg.augmentation.spec_augment,
                                    debug=DEBUG)

    valid_dataset = SpectrogramDataset(audio_conf           = model.audio_conf,
                                    manifest_filepath    = cfg.data.val_manifest,
                                    labels               = model.labels,
                                    normalize            = True,
                                    speed_volume_perturb = False,
                                    spec_augment         = False,
                                    debug=DEBUG)



    train_sampler = DSRandomSampler(dataset     = train_dataset,
                                    batch_size  = cfg.data.batch_size,
                                    start_index = state.training_step)

    # data load 하는 부분
    train_loader = AudioDataLoader(dataset       = train_dataset,
                                num_workers   = cfg.data.num_workers,
                                batch_sampler = train_sampler)

    test_loader = AudioDataLoader(dataset     = valid_dataset,
                                num_workers = cfg.data.num_workers,
                                batch_size  = cfg.data.batch_size)



    model      = model.to(device)
    parameters = model.parameters()

    if cfg.optimizer.adam:
            optimizer = torch.optim.AdamW(
                                        parameters,
                                        lr           = cfg.optimizer.learning_rate,
                                        betas        = cfg.optimizer.betas,
                                        eps          = cfg.optimizer.eps,
                                        weight_decay = cfg.optimizer.weight_decay
                                        )
    else:
        optimizer = torch.optim.SGD(
                                    parameters,
                                    lr           = cfg.optimizer.learning_rate,
                                    momentum     = cfg.optimizer.momentum,
                                    nesterov     = True,
                                    weight_decay = cfg.optimizer.weight_decay
                                    )



    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    criterion  = nn.CTCLoss(model.labels.index('_'), reduction='sum', zero_infinity=True)
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    if cfg.logger.useLogger == 'wandb':
        logger = WandBLogger(
                            projectTitle = cfg.logger.WBProjectTitle,
                            note         = cfg.logger.WBNote,
                            cfg          = dict(cfg)
                            )



    for epoch in range(state.epoch, cfg.training.epochs):
            model.train()
            end = time.time()
            start_epoch_time = time.time()
            state.set_epoch(epoch=epoch)
            train_sampler.set_epoch(epoch=epoch)
            train_sampler.reset_training_step(training_step=state.training_step)
            
            #train data있는거 가져다 사용하겠다.
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{cfg.training.epochs}", unit='batch') as pbar:
                for i, (data) in enumerate(train_loader, start=state.training_step):
                    state.set_training_step(training_step=i)
                    inputs, targets, input_percentages, target_sizes = data
                    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                    
                    # measure data loading time   
                    data_time.update(time.time() - end)
                    inputs = inputs.to(device)

                    out, output_sizes = model(inputs, input_sizes)

                    wer, cer, target_s, decoded_s = getCERandWER(evaluation_decoder, out, output_sizes, targets, target_sizes)

                    out = out.transpose(0, 1)  # TxNxH
                    float_out = out.float()  # ensure float32 for loss
                    float_out = float_out.log_softmax(-1) 


                    loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
                    loss = loss / inputs.size(0)  # average the loss by minibatch
                    loss_value = loss.item()

                    # Check to ensure valid loss was calculated
                    valid_loss, error = check_loss(loss, loss_value)
                    if valid_loss:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        print(error)
                        print('Skipping grad update')
                        loss_value = 0

                    state.avg_loss += loss_value
                    losses.update(loss_value, inputs.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    pbar.set_postfix({
                        'Time': f"{batch_time.val:.3f} ({batch_time.avg:.3f})",
                        'Data': f"{data_time.val:.3f} ({data_time.avg:.3f})",
                        'Loss': f"{losses.val:.4f} ({losses.avg:.4f})",
                        'WER': f"{wer:.4f}",
                        'CER': f"{cer:.4f}"
                    })
                    pbar.update(1)
                    
                    if cfg.logger.useLogger == 'wandb':
                        logger.update(losses)
                    

                    if cfg.checkpointing.checkpoint_per_iteration:
                        checkpoint_handler.save_iter_checkpoint_model(epoch=epoch, i=i, state=state)
                    del loss, out, float_out

                    if i % cfg.training.showResult == 0:
                        if not DEBUG:
                            print('target:\t',  target_s)
                            print('predicted:\t',decoded_s)

                    if i % cfg.training.clearCUDACash == 0:
                        # VRAM 초기화
                        torch.cuda.empty_cache()

            state.avg_loss /= (i + 1) # 수식 수정

            epoch_time = time.time() - start_epoch_time
            print('Training Summary Epoch: [{0}]\t'
                'Time taken (s): {epoch_time:.0f}\t'
                'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=state.avg_loss))

            if not DEBUG:       
                with torch.no_grad():
                    wer, cer, output_data = evaluate(test_loader=test_loader,
                                                    device=device,
                                                    model=model,
                                                    decoder=evaluation_decoder,
                                                    target_decoder=evaluation_decoder)

                state.add_results(epoch=epoch,
                                loss_result=state.avg_loss,
                                wer_result=wer,
                                cer_result=cer)

                print('Validation Summary Epoch: [{0}]\t'
                    'Average CER {cer:.3f}\t Average WER {wer:.3f}\t'.format(epoch + 1, cer=cer, wer=wer))


            if cfg.logger.useLogger == 'wandb':
                logger.update_valid(epoch, state.result_state)
            

            if cfg.checkpointing.checkpoint and not DEBUG:  # Save epoch checkpoint
                checkpoint_handler.save_checkpoint_model(epoch=epoch, state=state)
            # anneal lr
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / cfg.optimizer.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            if (state.best_cer is None or state.best_cer > cer):
                checkpoint_handler.save_best_model(epoch=epoch, state=state)
                state.set_best_cer(cer)
                state.reset_avg_loss()
            state.reset_training_step()  # Reset training step for next epoch



if __name__ == '__main__':
    args = parse_arguments()
    cfg = load_config(args.config)

    DEBUG = cfg.training.DEBUG

    if DEBUG:
        cfg.logger.useLogger = None
    train(cfg)

