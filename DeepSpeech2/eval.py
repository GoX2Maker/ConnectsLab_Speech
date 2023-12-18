import torch

import argparse
import yaml
from omegaconf import DictConfig

from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.testing import evaluate
from deepspeech_pytorch.utils import load_model

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


def eval(cfg):
    print(f'CUDA is available : {torch.cuda.is_available()}')
    print(f'CUDA Device Count  : {torch.cuda.device_count()}')
    device = torch.device(f"cuda:{cfg.cuda.cudaNumber}" if cfg.cuda.usingCuda else "cpu")
    torch.cuda.set_device(cfg.cuda.cudaNumber)
    print(f'Selected CUDA Number: {torch.cuda.current_device()}')


    model = load_model(device, cfg.model.savePath, cfg.model.half)
    decoder =GreedyDecoder(model.labels)
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                        manifest_filepath=cfg.data.test_manifest,
                                        labels=model.labels,
                                        normalize=True)
    test_loader = AudioDataLoader(test_dataset,
                                num_workers = cfg.data.num_workers,
                                batch_size  = cfg.data.batch_size)
    with torch.no_grad():
            wer, cer, output_data = evaluate(test_loader=test_loader,
                                            device=device,
                                            model=model,
                                            decoder=decoder,
                                            target_decoder=decoder,
                                            )

    print('Test Summary \t'
            'Average WER {wer:.3f}\t'
            'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))



if __name__ == '__main__':
    args = parse_arguments()
    cfg = load_config(args.config)
    eval(cfg)