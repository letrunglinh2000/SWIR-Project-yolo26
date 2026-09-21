import os
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG_DICT


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(
    r"D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project"
    r"\3.test\yolo26\NSLSR"
)

MODEL_PATH = Path(
    r"D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project"
    r"\3.test\yolo26\runs\train"
    r"\yolo26s_nslsr_pgd\weights\last.pt"
)

YAML_PATH = Path("dataset.yaml")

DEVICE = "0"
IMGSZ = 640

# PGD-20 test
PGD_EPS = 8.0 / 255.0
PGD_ALPHA = 2.0 / 255.0
PGD_STEPS = 20

# PGD is much more expensive than clean inference.
# Reduce this if GPU memory is insufficient.
BATCH_SIZE = 8

WORKERS = 4


# ============================================================
# DATASET SETUP
# ============================================================

def ensure_dataset(base_dir: Path, yaml_path: Path):

    print(f"Checking dataset: {base_dir}")

    for split in ["train", "val", "test"]:

        split_folder = base_dir / f"NSLSR_{split}"

        swir_dir = split_folder / "SWIR"
        images_dir = split_folder / "images"

        if not swir_dir.exists():
            continue

        # ----------------------------------------------------
        # YOLO wants images/ beside labels/
        # ----------------------------------------------------
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
                    f"Could not create junction ({e}). "
                    f"Using SWIR directly."
                )

        target_dir = (
            images_dir
            if images_dir.exists()
            else swir_dir
        )

        txt_path = base_dir / f"{split}.txt"

        img_list = sorted(
            target_dir.glob("*.jpg")
        )

        with open(txt_path, "w") as f:

            for img_path in img_list:

                f.write(
                    f"{os.path.abspath(img_path)}\n"
                )

        print(
            f"{split}: {len(img_list)} images"
        )

    # --------------------------------------------------------
    # dataset.yaml
    # --------------------------------------------------------

    yaml_content = f"""path: {base_dir.resolve().as_posix()}
train: train.txt
val: val.txt
test: test.txt

nc: 1
names: ['ship']
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset YAML: {yaml_path}")


# ============================================================
# PGD ATTACK
# ============================================================

def pgd_attack(
    model,
    batch,
    eps=8.0 / 255.0,
    alpha=2.0 / 255.0,
    steps=20,
    random_start=True,
):
    """
    Untargeted white-box L-inf PGD attack.

    Maximizes YOLO's native detection loss:

        max_delta L_YOLO(x + delta, y)

    subject to:

        ||delta||_inf <= eps

    The ground-truth labels and bounding boxes remain unchanged.
    """

    # Clean normalized images [0,1]
    x = batch["img"].detach()

    # --------------------------------------------------------
    # Random initialization inside epsilon-ball
    # --------------------------------------------------------

    if random_start:

        delta = torch.empty_like(x).uniform_(
            -eps,
            eps
        )

        x_adv = torch.clamp(
            x + delta,
            0.0,
            1.0
        )

    else:

        x_adv = x.clone()

    x_adv = x_adv.detach()

    # --------------------------------------------------------
    # YOLO loss expects training-mode detector outputs.
    #
    # But we do NOT want BatchNorm running statistics to change.
    # --------------------------------------------------------

    original_training_state = model.training

    bn_layers = [
        m
        for m in model.modules()
        if isinstance(
            m,
            nn.modules.batchnorm._BatchNorm
        )
    ]

    bn_states = [
        m.training
        for m in bn_layers
    ]

    model.train()

    # Freeze BN statistics
    for m in bn_layers:
        m.eval()

    try:

        for step in range(steps):

            x_adv.requires_grad_(True)

            # Same GT labels/boxes.
            # Only image is replaced.
            adv_batch = batch.copy()
            adv_batch["img"] = x_adv

            # -----------------------------------------------
            # Native YOLO detection loss
            # -----------------------------------------------

            output = model(adv_batch)

            if isinstance(output, (tuple, list)):
                loss = output[0]
            else:
                loss = output

            attack_loss = loss.sum()

            # -----------------------------------------------
            # Gradient w.r.t. IMAGE only
            # -----------------------------------------------

            grad = torch.autograd.grad(
                outputs=attack_loss,
                inputs=x_adv,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

            # -----------------------------------------------
            # PGD gradient ASCENT
            # -----------------------------------------------

            x_adv = (
                x_adv.detach()
                + alpha * grad.sign()
            )

            # -----------------------------------------------
            # Project into L-inf epsilon ball
            # -----------------------------------------------

            delta = torch.clamp(
                x_adv - x,
                min=-eps,
                max=eps
            )

            x_adv = torch.clamp(
                x + delta,
                min=0.0,
                max=1.0
            ).detach()

    finally:

        # Restore original model state
        model.train(original_training_state)

        for m, state in zip(
            bn_layers,
            bn_states
        ):
            m.train(state)

    return x_adv


# ============================================================
# PGD EVALUATION
# ============================================================

def evaluate_pgd(
    model_path,
    yaml_path,
    device_name="0",
    imgsz=640,
    batch_size=8,
    workers=4,
    eps=8.0 / 255.0,
    alpha=2.0 / 255.0,
    steps=20,
):

    print("\n==========================================")
    print(" YOLO26 PGD ADVERSARIAL TEST")
    print("==========================================")

    print(f"Model      : {model_path}")
    print(f"Epsilon    : {eps * 255:.1f}/255")
    print(f"Alpha      : {alpha * 255:.1f}/255")
    print(f"PGD steps  : {steps}")
    print(f"Image size : {imgsz}")
    print(f"Batch      : {batch_size}")

    print("==========================================\n")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = select_device(device_name)

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    yolo = YOLO(str(model_path))

    detector = yolo.model

    detector = detector.to(device)
    detector = detector.float()
    # ============================================================
# IMPORTANT FIX FOR YOLO26 LOSS
# ============================================================

# When loading a checkpoint directly for inference,
# detector.args may be a normal Python dict.
#
# YOLO26 detection loss expects:
#
#     model.args.box
#     model.args.cls
#     model.args.dfl
#
# instead of:
#
#     model.args["box"]
#
# Convert checkpoint arguments into Ultralytics' config Namespace.

    if isinstance(detector.args, dict):

        print(
            "Converting detector.args from dict "
            "to Ultralytics config Namespace..."
        )

        # Keep only valid Ultralytics config keys
        saved_args = {
            k: v
            for k, v in detector.args.items()
            if k in DEFAULT_CFG_DICT
        }

        # Merge with current defaults so required loss
        # hyperparameters are always available.
        detector.args = get_cfg(
            overrides=saved_args
        )


    # Force criterion to be recreated using the corrected args.
    detector.criterion = None


    print("YOLO loss hyperparameters:")
    print(f"  box = {detector.args.box}")
    print(f"  cls = {detector.args.cls}")
    print(f"  dfl = {detector.args.dfl}")
        
    detector.eval()

    # --------------------------------------------------------
    # We only require dL/dx.
    #
    # YOLO parameters themselves do not need gradients.
    # This considerably reduces PGD memory usage.
    # --------------------------------------------------------

    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    # --------------------------------------------------------
    # Create normal Ultralytics DetectionValidator
    #
    # We use it for:
    #   - dataloader
    #   - letterbox preprocessing
    #   - target handling
    #   - NMS
    #   - mAP
    # --------------------------------------------------------

    validator = DetectionValidator(
        args={
            "model": str(model_path),
            "data": str(yaml_path.resolve()),
            "split": "test",

            "imgsz": imgsz,
            "batch": batch_size,
            "device": device_name,
            "workers": workers,

            "rect": True,

            # Standard YOLO validation settings
            "conf": 0.001,
            "iou": 0.7,

            "plots": False,
            "save_json": False,
            "save_txt": False,

            "verbose": True,
            "mode": "val",
        }
    )

    # Standalone validation
    validator.training = False

    validator.device = device

    # Force float32 attack
    validator.args.quantize = None

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    validator.data = check_det_dataset(
        str(yaml_path.resolve()),
        split="test"
    )

    # YOLO maximum stride, normally 32
    validator.stride = int(
        detector.stride.max().item()
    )

    validator.dataloader = validator.get_dataloader(
        validator.data["test"],
        batch_size
    )

    # Initialize metric system
    validator.init_metrics(detector)

    num_batches = len(
        validator.dataloader
    )

    global_max_delta = 0.0
    global_mean_delta = 0.0

    # ========================================================
    # ADVERSARIAL TEST LOOP
    # ========================================================

    for batch_idx, batch in enumerate(
        validator.dataloader
    ):

        # ----------------------------------------------------
        # Standard YOLO test preprocessing
        #
        # image:
        # uint8 [0,255]
        #       ↓
        # float [0,1]
        # ----------------------------------------------------

        batch = validator.preprocess(batch)

        clean_img = (
            batch["img"]
            .detach()
            .clone()
        )

        # ----------------------------------------------------
        # PGD attack
        # ----------------------------------------------------

        adv_img = pgd_attack(
            model=detector,
            batch=batch,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=True,
        )

        # ----------------------------------------------------
        # Attack sanity check
        # ----------------------------------------------------

        delta = (
            adv_img - clean_img
        ).abs()

        max_delta = (
            delta.max().item()
        )

        mean_delta = (
            delta.mean().item()
        )

        global_max_delta = max(
            global_max_delta,
            max_delta
        )

        global_mean_delta += mean_delta

        # Replace clean input by attacked input
        batch["img"] = adv_img

        # ----------------------------------------------------
        # Normal YOLO inference on adversarial image
        # ----------------------------------------------------

        detector.eval()

        with torch.inference_mode():

            preds = detector(
                batch["img"]
            )

            preds = validator.postprocess(
                preds
            )

        # ----------------------------------------------------
        # Standard YOLO metric calculation
        # ----------------------------------------------------

        validator.update_metrics(
            preds,
            batch
        )

        print(
            f"\rPGD batch "
            f"{batch_idx + 1}/{num_batches}"
            f" | max |delta|="
            f"{max_delta * 255:.2f}/255",
            end=""
        )

    print()

    # ========================================================
    # FINAL METRICS
    # ========================================================

    validator.gather_stats()

    stats = validator.get_stats()

    validator.finalize_metrics()

    print("\n==========================================")
    print(" PGD TEST RESULTS")
    print("==========================================")

    validator.print_results()

    print("\nDetailed metrics:")

    for key, value in stats.items():

        if isinstance(
            value,
            (int, float)
        ):
            print(
                f"{key:30s}: "
                f"{value:.6f}"
            )

    global_mean_delta /= max(
        num_batches,
        1
    )

    print("\nAttack verification:")

    print(
        f"Maximum perturbation : "
        f"{global_max_delta * 255:.4f}/255"
    )

    print(
        f"Mean |perturbation|  : "
        f"{global_mean_delta * 255:.4f}/255"
    )

    print(
        f"Allowed epsilon      : "
        f"{eps * 255:.4f}/255"
    )

    # Safety assertion:
    # PGD must NEVER exceed epsilon.
    assert (
        global_max_delta
        <= eps + 1e-5
    ), (
        "ERROR: PGD perturbation exceeded "
        "the epsilon constraint."
    )

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    ensure_dataset(
        BASE_DIR,
        YAML_PATH
    )

    # --------------------------------------------------------
    # 2. Model check
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print(
            f"ERROR: Model not found:\n"
            f"{MODEL_PATH}"
        )

        return

    # --------------------------------------------------------
    # 3. PGD evaluation
    # --------------------------------------------------------

    evaluate_pgd(
        model_path=MODEL_PATH,
        yaml_path=YAML_PATH,

        device_name=DEVICE,

        imgsz=IMGSZ,
        batch_size=BATCH_SIZE,
        workers=WORKERS,

        eps=PGD_EPS,
        alpha=PGD_ALPHA,
        steps=PGD_STEPS,
    )


if __name__ == "__main__":
    main()

# test image x
#      │
#      │  GT box + class
#      ↓
# PGD-20
#      │
#      │ maximize YOLO detection loss
#      │
#      │ ∂L / ∂x
#      ↓
# x_adv = x + δ
#      │
#      │ ||δ||∞ ≤ 8/255
#      ↓
# YOLO26
#      ↓
# prediction
#      ↓
# mAP_PGD