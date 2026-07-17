"""Quick executable check for the DA-TWML task-weighting logic.

This script intentionally avoids loading the petrographic image dataset. It
checks the scoring, Top-k selection, and softmax-weighted meta-loss computation
on synthetic episode-level values so that reviewers can verify the repository
quickly after installation.
"""

from __future__ import annotations

import torch


def normalize(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if float(std) < 1e-12:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def main() -> None:
    torch.manual_seed(7)

    query_loss = torch.tensor([0.42, 0.81, 0.35, 0.67], dtype=torch.float32)
    uncertainty = torch.tensor([0.31, 0.73, 0.28, 0.61], dtype=torch.float32)
    domain_gap = torch.tensor([0.12, 0.55, 0.19, 0.44], dtype=torch.float32)
    prototype_confusion = torch.tensor([0.22, 0.64, 0.25, 0.58], dtype=torch.float32)

    score = (
        normalize(query_loss)
        + normalize(uncertainty)
        + normalize(domain_gap)
        + normalize(prototype_confusion)
    )

    topk = 2
    selected = torch.topk(score, k=topk).indices
    weights = torch.softmax(score[selected], dim=0)
    weighted_meta_loss = torch.sum(weights * query_loss[selected])

    assert selected.numel() == topk
    assert torch.isfinite(weighted_meta_loss)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert selected.tolist() == [1, 3]

    print("DA-TWML quick test passed.")
    print(f"Selected episodes: {selected.tolist()}")
    print(f"Weighted meta-loss: {weighted_meta_loss.item():.6f}")


if __name__ == "__main__":
    main()

