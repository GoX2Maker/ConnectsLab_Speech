import argparse
import yaml
from omegaconf import DictConfig
import ast

import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from parts.model import AudioClassificationModel
from parts.loader import AudioDataset

from sklearn.model_selection import train_test_split


def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        cfg = DictConfig(yaml.safe_load(file))
        return cfg

def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepSpeech training script')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration yaml file')
    # parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides yaml config if provided)')
    # Add other command-line arguments as needed
    return parser.parse_args()

def checkFileExists(audioPaths):
    for file in audioPaths:
        if not os.path.exists(file):
            raise Exception("Check audio path")
        
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_model(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    # Wrap loader with tqdm for a progress bar
    eval_loop = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for inputs, labels in eval_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            predicted = outputs.squeeze().round()

            # Update correct and total counts
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def eval(cfg):
    seed_everything(cfg.training.seed)

    print(f'CUDA is available : {torch.cuda.is_available()}')
    print(f'CUDA Device Count  : {torch.cuda.device_count()}')
    device = torch.device(f"cuda:{cfg.training.cudaNumber}" if cfg.training.useCuda else "cpu")
    torch.cuda.set_device(cfg.training.cudaNumber)
    print(f'Selected CUDA Number: {torch.cuda.current_device()}')

    # Test
    df_test = pd.read_csv(cfg.data.testDataPath)
    audioPaths_test = df_test['file_name']
    labels_test = df_test['gender']
    audioPaths_test, _, labels_test, _ = train_test_split(
    audioPaths_test, labels_test, 
    train_size=cfg.data.testSize,  # Size of the subset
    stratify=labels_test,  # Stratify based on labels to maintain ratio
    random_state=cfg.training.seed  # For reproducibility
)
    

    audioPaths_test = audioPaths_test.reset_index(drop=True)
    labels_test = labels_test.reset_index(drop=True)

    print("Check audioPaths_valid")
    checkFileExists(audioPaths_test)
    print("OK")


    test_dataset = AudioDataset(audioPaths = audioPaths_test,
                        labels = labels_test,
                        device=device,
                        n_mfcc = cfg.data.n_mfcc,
                        sample_rate = cfg.data.sample_rate,
                        window_size = cfg.data.window_size,
                        hop_length = cfg.data.hop_length,
                        desired_frames = cfg.data.desired_frames,
                        )
    
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    
        # check Loader
    if cfg.data.loaderCheck:
        for loader_name, loader in zip(["Test"], [test_dataset]):
            print(f"Checking {loader_name} DataLoader...")
            try:
                # Wrap the DataLoader with tqdm for a progress bar
                for i, data in enumerate(tqdm(loader, desc=f"Processing {loader_name} DataLoader")):
                    # Replace 'data' with actual unpacking of your batch data, e.g., inputs, labels = data
                    pass  # Process your data here
                print(f"{loader_name} DataLoader is fine!")
            except Exception as e:
                print(f"Error in {loader_name} DataLoader at batch index {i}: {e}")
                break  # Stop checking after the first error
    

# Ensure your model architecture is defined
    model = AudioClassificationModel()

    # Load the saved model's state
    model.load_state_dict(torch.load(os.path.join(cfg.checkpointing.save_model)))
    model.to(device)


    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy}")


if __name__ == '__main__':
    args = parse_arguments()
    cfg = load_config(args.config)

    eval(cfg)
