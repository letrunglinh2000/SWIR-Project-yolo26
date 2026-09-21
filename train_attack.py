import os
import argparse
from pathlib import Path
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model


# ============================================================
# PGD ADVERSARIAL TRAINER
# ============================================================

@contextmanager
def freeze_batchnorm_stats(model):
    """
    Prevent PGD's extra forward passes from repeatedly updating
    BatchNorm running statistics.

    Important:
    - YOLO itself stays in training mode.
    - Only BatchNorm running statistics are temporarily frozen.
    """
    bn_layers = [
        m for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]

    states = [m.training for m in bn_layers]

    try:
        for m in bn_layers:
            m.eval()
        yield
    finally:
        for m, state in zip(bn_layers, states):
            m.train(state)


class PGDDetectionTrainer(DetectionTrainer):
    """
    YOLO26 DetectionTrainer with on-the-fly PGD adversarial training.

    Values are configured in train() before model.train() is called.

    adv_eps / adv_alpha are already normalized to [0,1].
    """

    adv_eps = 8.0 / 255.0
    adv_alpha = 2.0 / 255.0
    adv_steps = 3
    adv_ratio = 0.5
    adv_start_epoch = 0
    adv_random_start = True

    def generate_pgd(self, batch):
        """
        Generate an L-infinity PGD adversarial batch by maximizing
        YOLO's native detection loss.

        Original:
            x

        Adversarial:
            x_adv = x + delta

        subject to:
            ||delta||_inf <= epsilon
        """

        x = batch["img"].detach()

        # ----------------------------------------------------
        # Initial adversarial image
        # ----------------------------------------------------
        if self.adv_random_start:
            delta = torch.empty_like(x).uniform_(
                -self.adv_eps,
                self.adv_eps
            )

            x_adv = torch.clamp(
                x + delta,
                min=0.0,
                max=1.0
            )
        else:
            x_adv = x.clone()

        x_adv = x_adv.detach()

        # Use unwrapped YOLO model.
        # This avoids unnecessary DDP gradient synchronization
        # while constructing adversarial examples.
        attack_model = unwrap_model(self.model)

        # ----------------------------------------------------
        # PGD iterations
        # ----------------------------------------------------
        with freeze_batchnorm_stats(attack_model):

            for _ in range(self.adv_steps):

                x_adv.requires_grad_(True)

                # Same labels / bounding boxes,
                # only replace input image.
                adv_batch = batch.copy()
                adv_batch["img"] = x_adv

                # PGD is more numerically reliable in FP32.
                amp_context = (
                    torch.autocast(
                        device_type="cuda",
                        enabled=False
                    )
                    if self.device.type == "cuda"
                    else nullcontext()
                )

                with torch.enable_grad(), amp_context:

                    # YOLO batch input -> native detection loss
                    output = attack_model(adv_batch)

                    # Current Ultralytics normally returns:
                    # (loss, loss_items)
                    if isinstance(output, (tuple, list)):
                        loss = output[0]
                    else:
                        loss = output

                    attack_loss = loss.sum()

                    # Gradient ONLY with respect to input.
                    # This does NOT update YOLO weights.
                    grad = torch.autograd.grad(
                        attack_loss,
                        x_adv,
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True
                    )[0]

                # ------------------------------------------------
                # Gradient ASCENT:
                # maximize detection loss
                # ------------------------------------------------
                x_adv = (
                    x_adv.detach()
                    + self.adv_alpha * grad.sign()
                )

                # ------------------------------------------------
                # Project perturbation back into epsilon-ball
                # ------------------------------------------------
                delta = torch.clamp(
                    x_adv - x,
                    min=-self.adv_eps,
                    max=self.adv_eps
                )

                # Image must remain valid [0,1]
                x_adv = torch.clamp(
                    x + delta,
                    min=0.0,
                    max=1.0
                ).detach()

        return x_adv


    def preprocess_batch(self, batch):
        """
        Standard YOLO preprocessing
        +
        PGD adversarial example generation.
        """

        # ----------------------------------------------------
        # Standard Ultralytics preprocessing
        #
        # uint8 -> float32
        # [0,255] -> [0,1]
        # device transfer
        # multi-scale resizing if enabled
        # ----------------------------------------------------
        batch = super().preprocess_batch(batch)

        # ----------------------------------------------------
        # Allow normal training before adversarial training
        # ----------------------------------------------------
        if self.epoch < self.adv_start_epoch:
            return batch

        if self.adv_ratio <= 0.0:
            return batch

        if self.adv_steps <= 0:
            return batch

        clean_img = batch["img"].detach()

        # Generate attack using current YOLO model
        adv_img = self.generate_pgd(batch)

        # ----------------------------------------------------
        # Select how much of each batch is adversarial
        #
        # adv_ratio = 0.0 -> 100% clean
        # adv_ratio = 0.5 -> 50% clean + 50% adversarial
        # adv_ratio = 1.0 -> 100% adversarial
        # ----------------------------------------------------
        if self.adv_ratio >= 1.0:

            batch["img"] = adv_img

        else:

            batch_size = clean_img.shape[0]

            use_adv = (
                torch.rand(
                    batch_size,
                    device=clean_img.device
                )
                < self.adv_ratio
            )

            # [B] -> [B,1,1,1]
            use_adv = use_adv.view(
                batch_size,
                1,
                1,
                1
            )

            batch["img"] = torch.where(
                use_adv,
                adv_img,
                clean_img
            )

        return batch


# ============================================================
# DATASET SETUP
# ============================================================

def ensure_dataset(base_dir: Path, yaml_path: Path):
    """
    Ensure the dataset directory structure, junctions,
    and txt path files exist.

    YOLO expects an 'images' folder alongside 'labels'
    to auto-resolve annotations.
    """

    base_dir = Path(base_dir)

    print(f"Checking dataset in: {base_dir}")

    for split in ["train", "val", "test"]:

        split_folder = base_dir / f"NSLSR_{split}"

        swir_dir = split_folder / "SWIR"
        images_dir = split_folder / "images"

        if swir_dir.exists():

            # ------------------------------------------------
            # Create Windows directory junction
            # ------------------------------------------------
            if not images_dir.exists():

                try:
                    import _winapi

                    _winapi.CreateJunction(
                        str(swir_dir.resolve()),
                        str(images_dir.resolve())
                    )

                    print(
                        f"Created junction: "
                        f"{images_dir} -> {swir_dir}"
                    )

                except Exception as e:

                    print(
                        f"Notice: Could not create junction "
                        f"({e}), using direct SWIR path."
                    )

            target_dir = (
                images_dir
                if images_dir.exists()
                else swir_dir
            )

            txt_path = base_dir / f"{split}.txt"

            # ------------------------------------------------
            # Generate image list
            # ------------------------------------------------
            img_list = sorted(
                list(target_dir.glob("*.jpg"))
            )

            with open(txt_path, "w") as f:

                for img_path in img_list:

                    f.write(
                        f"{os.path.abspath(img_path)}\n"
                    )

            print(
                f"Ready: {txt_path} "
                f"({len(img_list)} images)"
            )

    # --------------------------------------------------------
    # dataset.yaml
    # --------------------------------------------------------
    if not yaml_path.exists():

        yaml_content = f"""path: {base_dir.resolve().as_posix()}
train: train.txt
val: val.txt
test: test.txt

nc: 1
names: ['ship']
"""

        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(
            f"Created dataset configuration: "
            f"{yaml_path}"
        )


# ============================================================
# TRAIN
# ============================================================

def train(args):

    # --------------------------------------------------------
    # Validate adversarial parameters
    # --------------------------------------------------------

    if args.adv_eps < 0:
        raise ValueError("--adv-eps must be >= 0")

    if args.adv_alpha <= 0:
        raise ValueError("--adv-alpha must be > 0")

    if args.adv_steps < 0:
        raise ValueError("--adv-steps must be >= 0")

    if not 0.0 <= args.adv_ratio <= 1.0:
        raise ValueError(
            "--adv-ratio must be between 0 and 1"
        )

    # --------------------------------------------------------
    # 1. Dataset verification
    # --------------------------------------------------------

    base_dir = Path(
        r"D:\letrunglinh\1.Paper_IVCL"
        r"\11.SWIR_Project\3.test"
        r"\yolo26\NSLSR"
    )

    yaml_path = Path(args.data)

    ensure_dataset(
        base_dir,
        yaml_path
    )

    # --------------------------------------------------------
    # 2. Check model
    # --------------------------------------------------------

    model_path = args.model

    if not os.path.exists(model_path):

        print(
            f"\n[ERROR] Model file "
            f"'{model_path}' not found!"
        )

        print(
            "Please check the model path or place "
            "'yolo26n.pt' in the working directory."
        )

        return

    # --------------------------------------------------------
    # Configure PGD trainer
    #
    # CLI uses pixel scale:
    #
    # --adv-eps 8
    #
    # means:
    #
    # epsilon = 8/255
    # --------------------------------------------------------

    PGDDetectionTrainer.adv_eps = (
        args.adv_eps / 255.0
    )

    PGDDetectionTrainer.adv_alpha = (
        args.adv_alpha / 255.0
    )

    PGDDetectionTrainer.adv_steps = (
        args.adv_steps
    )

    PGDDetectionTrainer.adv_ratio = (
        args.adv_ratio
    )

    PGDDetectionTrainer.adv_start_epoch = (
        args.adv_start_epoch
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("\n==========================================")
    print(" YOLO26 ADVERSARIAL TRAINING")
    print("==========================================")

    print(f" Model       : {model_path}")
    print(f" Data config : {yaml_path}")

    print(f" Epochs      : {args.epochs}")
    print(f" Batch size  : {args.batch}")
    print(f" Image size  : {args.imgsz}")

    print(f" Device      : {args.device}")
    print(f" Workers     : {args.workers}")

    print(
        f" Project/Name: "
        f"{args.project}/{args.name}"
    )

    print("------------------------------------------")

    print(" Attack      : PGD L-inf")

    print(
        f" Epsilon     : "
        f"{args.adv_eps}/255"
    )

    print(
        f" Step size   : "
        f"{args.adv_alpha}/255"
    )

    print(
        f" PGD steps   : "
        f"{args.adv_steps}"
    )

    print(
        f" Adv ratio   : "
        f"{args.adv_ratio:.2f}"
    )

    print(
        f" Start epoch : "
        f"{args.adv_start_epoch}"
    )

    print("==========================================\n")

    # --------------------------------------------------------
    # 3. Initialize YOLO26
    # --------------------------------------------------------

    model = YOLO(model_path)

    # --------------------------------------------------------
    # 4. Train
    # --------------------------------------------------------

    proj_dir = (
        str(Path(args.project).resolve())
        if args.project
        else None
    )

    results = model.train(

        trainer=PGDDetectionTrainer,

        data=str(yaml_path.resolve()),

        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,

        device=args.device,
        workers=args.workers,

        optimizer=args.optimizer,
        lr0=args.lr0,

        patience=args.patience,

        save=True,

        project=proj_dir,
        name=args.name,

        exist_ok=args.exist_ok,

        pretrained=args.pretrained,
        resume=args.resume,

        verbose=True,
    )

    print(
        "\nAdversarial training "
        "completed successfully!"
    )

    save_dir = (
        Path(results.save_dir)
        if hasattr(results, "save_dir")
        else Path(args.project) / args.name
    )

    print(
        "Model weights saved to: "
        f"{save_dir / 'weights' / 'best.pt'}"
    )

    return results


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Adversarially train YOLO26 "
            "on NSLSR SWIR Dataset"
        )
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Initial YOLO26 weights"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="dataset.yaml"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--device",
        default="0"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/train"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolo26n_nslsr_pgd"
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False
    )

    # ========================================================
    # ADVERSARIAL TRAINING PARAMETERS
    # ========================================================

    parser.add_argument(
        "--adv-eps",
        type=float,
        default=8.0,
        help=(
            "PGD epsilon in 0-255 pixel scale. "
            "8 means 8/255."
        )
    )

    parser.add_argument(
        "--adv-alpha",
        type=float,
        default=2.0,
        help=(
            "PGD step size in 0-255 pixel scale. "
            "2 means 2/255."
        )
    )

    parser.add_argument(
        "--adv-steps",
        type=int,
        default=3,
        help="Number of PGD attack iterations"
    )

    parser.add_argument(
        "--adv-ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of each batch replaced "
            "by adversarial images"
        )
    )

    parser.add_argument(
        "--adv-start-epoch",
        type=int,
        default=0,
        help=(
            "Start adversarial training at "
            "this epoch"
        )
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    train(args)


# python train_attack.py --model yolo26s.pt --name yolo26s_nslsr_pgd --adv-eps 8 --adv-alpha 2 --adv-steps 3 --adv-ratio 0.5



        #          ORIGINAL YOLO BATCH
        #                 x
        #                 │
        #      standard augmentation
        #                 │
        #                 ▼
        #          preprocess_batch
        #          normalize [0,1]
        #                 │
        #       ┌─────────┴──────────┐
        #       │                    │
        #    CLEAN x             PGD attack
        #                            │
        #                  maximize YOLO loss
        #                            │
        #                  ∂Ldet / ∂x
        #                            │
        #                            ▼
        #                      x_adv = x + δ
        #       │                    │
        #       │                    │
        #       └─────────┬──────────┘
        #                 │
        #             50% / 50%
        #                 │
        #                 ▼
        #              YOLO26
        #                 │
        #                 ▼
        #        detection training loss
        #                 │
        #                 ▼
        #            backward()
        #                 │
        #                 ▼
        #         update YOLO weights