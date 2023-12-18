import torch


import os 

import argparse
import yaml
from omegaconf import DictConfig

from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.utils import load_model
from deepspeech_pytorch.inference import transcribe

def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepSpeech training script')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration yaml file')
    # parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides yaml config if provided)')
    # Add other command-line arguments as needed
    return parser.parse_args()

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        cfg = DictConfig(yaml.safe_load(file))
        return cfg

def trans_(path, spect_parser, model, decoder, device):
    decoded_output, decoded_offsets, out, output_sizes = transcribe(audio_path=path,
                                                spect_parser=spect_parser,
                                                model=model,
                                                decoder=decoder,
                                                device=device,
                                                use_half=False)
    
    print(path, decoded_output[0][0])



def trans(cfg):
    print(f'CUDA is available : {torch.cuda.is_available()}')
    print(f'CUDA Device Count  : {torch.cuda.device_count()}')
    device = torch.device(f"cuda:{cfg.cuda.cudaNumber}" if cfg.cuda.usingCuda else "cpu")
    torch.cuda.set_device(cfg.cuda.cudaNumber)
    print(f'Selected CUDA Number: {torch.cuda.current_device()}')




    model = load_model(device, cfg.model.savePath, cfg.model.half)
    decoder =GreedyDecoder(model.labels)
    spect_parser = SpectrogramParser(audio_conf=model.audio_conf,
                                        normalize=True)

    with torch.no_grad():
        model.eval()

        if cfg.data.readCNT > 0:
            cnt =0
            paths = os.listdir(cfg.data.audioFolder)
            for path in paths:
                if 'wav' in path:
                    path = os.path.join(cfg.data.audioFolder, path)
                    
                    trans_(path, spect_parser, model, decoder, device)
                    cnt += 1
                    if cnt == cfg.data.readCNT:
                        break
        else:
            for path in os.listdir(cfg.data.audioFolder):
                if 'wav' in path:
                    path = os.path.join(cfg.data.audioFolder, path)
                    trans_(path, spect_parser, model, decoder, device)
                

if __name__ == '__main__':
    args = parse_arguments()
    cfg = load_config(args.config)
    trans(cfg)
