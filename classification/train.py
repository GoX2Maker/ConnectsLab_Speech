import argparse
import yaml
from omegaconf import DictConfig
import ast

import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from parts.model import AudioClassificationModel
from parts.loader import AudioDataset

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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def checkFileExists(audioPaths):
    for file in audioPaths:
        if not os.path.exists(file):
            raise Exception("Check audio path")
        
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


def train(cfg):
    seed_everything(cfg.training.seed)

    print(f'CUDA is available : {torch.cuda.is_available()}')
    print(f'CUDA Device Count  : {torch.cuda.device_count()}')
    device = torch.device(f"cuda:{cfg.training.cudaNumber}" if cfg.training.useCuda else "cpu")
    torch.cuda.set_device(cfg.training.cudaNumber)
    print(f'Selected CUDA Number: {torch.cuda.current_device()}')


    df_train = pd.read_csv(cfg.data.trainDataPath)
    audioPaths_train = df_train['file_name']
    labels_train = df_train['gender']

    # Train, valid
    # Assuming audioPaths_train and labels_train are already defined
    audioPaths_train, _, labels_train, _ = train_test_split(
        audioPaths_train, labels_train, 
        train_size=cfg.data.trainSize,  # Size of the subset
        stratify=labels_train,  # Stratify based on labels to maintain ratio
        random_state=cfg.training.seed  # For reproducibility
    )

    audioPaths_train, audioPaths_valid, labels_train, labels_valid = train_test_split(audioPaths_train, labels_train, test_size=cfg.data.validRate, shuffle=True, random_state=cfg.training.seed) 


    audioPaths_train = audioPaths_train.reset_index(drop=True)
    audioPaths_valid = audioPaths_valid.reset_index(drop=True)

    labels_train = labels_train.reset_index(drop=True)
    labels_valid = labels_valid.reset_index(drop=True)


    print("Check audioPaths_train")
    checkFileExists(audioPaths_train)
    print("OK")

    print("Check audioPaths_valid")
    checkFileExists(audioPaths_valid)
    print("OK")

    train_dataset = AudioDataset(audioPaths = audioPaths_train,
                        labels = labels_train,
                        device=device,
                        n_mfcc = cfg.data.n_mfcc,
                        sample_rate = cfg.data.sample_rate,
                        window_size = cfg.data.window_size,
                        hop_length = cfg.data.hop_length,
                        desired_frames = cfg.data.desired_frames,
                        )

    valid_dataset = AudioDataset(audioPaths = audioPaths_valid,
                            labels = labels_valid,
                            device=device,
                            n_mfcc = cfg.data.n_mfcc,
                            sample_rate = cfg.data.sample_rate,
                            window_size = cfg.data.window_size,
                            hop_length = cfg.data.hop_length,
                            desired_frames = cfg.data.desired_frames,
                            )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False)


    # check Loader
    if cfg.data.loaderCheck:
        for loader_name, loader in zip(["Train", "Validation"], [train_loader, valid_loader]):
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

    model = AudioClassificationModel()
    model.to(device)

    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.AdamW(    model.parameters(),
                                lr           = cfg.optimizer.learning_rate,
                                betas        = cfg.optimizer.betas,
                                eps          = cfg.optimizer.eps,
                                weight_decay = cfg.optimizer.weight_decay
                            )
    
    best_valid_accuracy = 0.0  # Initialize best validation accuracy

    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Wrap train_loader with tqdm for a progress bar
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loop.set_postfix(loss=total_loss/len(train_loader))

        print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Loss: {total_loss / len(train_loader)}")
        
        # Evaluate on validation set and print with tqdm
        valid_accuracy = evaluate_model(model, valid_loader,device=device)
        tqdm.write(f"Validation Accuracy: {valid_accuracy}")

        # Save the best model
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            # Define the model save path
            save_path = os.path.join(cfg.checkpointing.save_folder, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"Saved Best Model with Accuracy: {best_valid_accuracy}")


if __name__ == '__main__':
    args = parse_arguments()
    cfg = load_config(args.config)

    train(cfg)
