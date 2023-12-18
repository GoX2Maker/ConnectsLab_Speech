import argparse
import yaml
from omegaconf import DictConfig

import os


import torch


from parts.model import AudioClassificationModel
from parts.loader import AudioTransform


def load_config(yaml_file_path):
    with open(yaml_file_path, "r") as file:
        cfg = DictConfig(yaml.safe_load(file))
        return cfg


def parse_arguments():
    parser = argparse.ArgumentParser(description="DeepSpeech training script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration yaml file"
    )
    # parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides yaml config if provided)')
    # Add other command-line arguments as needed
    return parser.parse_args()


def convertSEX(output):
    output = output.round()
    if output[0][0] == 0:
        output = "남"
    else:
        output = "여"
    return output


def trans(cfg):
    print(f"CUDA is available : {torch.cuda.is_available()}")
    print(f"CUDA Device Count  : {torch.cuda.device_count()}")
    device = torch.device(
        f"cuda:{cfg.training.cudaNumber}" if cfg.training.useCuda else "cpu"
    )
    torch.cuda.set_device(cfg.training.cudaNumber)
    print(f"Selected CUDA Number: {torch.cuda.current_device()}")

    # Ensure your model architecture is defined
    model = AudioClassificationModel()

    # Load the saved model's state
    model.load_state_dict(torch.load(os.path.join(cfg.checkpointing.save_model)))

    # Load the saved model's    state
    model.to(device)

    transform = AudioTransform(
        device=device,
        n_mfcc=cfg.data.n_mfcc,
        sample_rate=cfg.data.sample_rate,
        window_size=cfg.data.window_size,
        hop_length=cfg.data.hop_length,
        desired_frames=cfg.data.desired_frames,
    )

    with torch.no_grad():
        model.eval()

        if cfg.data.readCNT > 0:
            cnt = 0
            paths = os.listdir(cfg.data.audioFolder)
            for path in paths:
                if "wav" in path:
                    path = os.path.join(cfg.data.audioFolder, path)

                    mfcc = transform.getMFCC(path)
                    output = convertSEX(model(mfcc))

                    print(path, output)

                    cnt += 1
                    if cnt == cfg.data.readCNT:
                        break
        else:
            for path in os.listdir(cfg.data.audioFolder):
                if "wav" in path:
                    path = os.path.join(cfg.data.audioFolder, path)
                    mfcc = transform.getMFCC(path).unsqueeze(1)

                    output = convertSEX(model(mfcc))

                    print(path, output)


if __name__ == "__main__":
    args = parse_arguments()
    cfg = load_config(args.config)

    trans(cfg)
