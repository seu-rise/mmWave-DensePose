"""Run a finite-output check for a public paper model.

    python -m example.models.forward --model PASTNet
    python -m example.models.forward --all --device cpu
    python -m example.models.forward --model PASTNet --backward
"""

import argparse
import json

import torch

from mmwave_densepose.models import build_model, list_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list_models(), default="PASTNet")
    parser.add_argument("--all", action="store_true", help="Check all paper models.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--points", type=int, default=300)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--input-dim", type=int, choices=(3, 4, 5), default=5)
    parser.add_argument("--ablation", choices=("full", "spatial_only", "baseline"), default="full")
    parser.add_argument("--backward", action="store_true", help="Also check gradients from a synthetic regression target.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = {"input_dim": args.input_dim, "num_points": args.points,
              "seq_len": args.frames, "ablation": args.ablation}
    for name in list_models() if args.all else [args.model]:
        model = build_model(name, config).to(args.device)
        model.train(args.backward)
        points = torch.rand(2, args.frames, args.points, args.input_dim, device=args.device)
        with torch.set_grad_enabled(args.backward):
            if name == "MMPoseFK" and args.backward:
                details = model(points, return_dict=True)
                output = details["pred_visual_target"]
            else:
                output = model(points)
            if not torch.isfinite(output).all():
                raise RuntimeError(f"{name} produced non-finite predictions.")
            if name != "PCCCompletion" and output.shape != (2, 19, 3):
                raise RuntimeError(f"{name} returned unexpected output shape: {output.shape}")
            if args.backward:
                if name == "MMPoseFK":
                    batch = {"pose_target": torch.rand_like(output)}
                    loss_fn = torch.nn.MSELoss()
                    loss = model.compute_target_loss(details, batch, loss_fn)
                    loss += sum(model.compute_auxiliary_losses(details, batch, loss_fn).values())
                else:
                    loss = output.square().mean()
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError(f"{name} produced missing or non-finite gradients.")
        print(json.dumps({"model": name, "input_shape": list(points.shape),
                          "output_shape": list(output.shape), "finite": True,
                          "backward": args.backward}), flush=True)
        del model, output, points


if __name__ == "__main__":
    main()
