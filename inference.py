"""
ArcFace Model Inference Script
"""

import argparse
import os
import torch

from backbones import get_model
from eval import verification


@torch.no_grad()
def inference(weight_path, network_name):
    """Load model and evaluate on benchmarks."""

    # Load model
    net = get_model(network_name, fp16=False)
    net.load_state_dict(torch.load(weight_path))
    net.eval()

    # Evaluation benchmarks
    benchmarks = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    base_path = ""

    for benchmark in benchmarks:
      
        print(f"{'#' * 20} {benchmark.upper()} {'#' * 20}")

        bin_path = os.path.join(base_path, f"{benchmark}.bin")
        dataset = verification.load_bin(bin_path, image_size=(112, 112))

        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
            dataset, net, 10, 10
        )

        print(f"Accuracy: {acc1:.4f} ± {std1:.4f}")
        print(f"Feature norm: {xnorm:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArcFace Model Inference')
    parser.add_argument('--network', type=str, default='r50',
                        help='backbone network')
    parser.add_argument('--weight', type=str, required=True,
                        help='path to model weights')

    args = parser.parse_args()
    inference(args.weight, args.network)