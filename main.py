import fire
import torch
import copy
import pdb
from scripts import utils
from experiments.ablation import corrupt_inference

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 2500,
    max_gen_len: int = 500,
    max_batch_size: int = 8,
    model_parallel_size = None
):
    """ 
    Each experiments are maintained in their own scripts. Main.py is used to demonstrate following functionalities:
    1. Load a model (Llama) and tokenizer
    2. Run inference on the Llama
    3. Call individual experiments
    """
    
    corrupt_inference.main(ckpt_dir, tokenizer_path, max_seq_len, max_gen_len, max_batch_size, model_parallel_size)

if __name__ == '__main__':
    fire.Fire(main)