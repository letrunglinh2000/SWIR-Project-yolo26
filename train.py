import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def ensure_dataset(base_dir: Path, yaml_path: Path):
    """
    Ensure the dataset directory structure, junctions, and txt path files exist.
    YOLO expects an 'images' folder alongside 'labels' to auto-resolve annotations.
    """
    base_dir = Path(base_dir)
    print(f"Checking dataset in: {base_dir}")

    for split in ['train', 'val', 'test']:
        split_folder = base_dir / f'NSLSR_{split}'
        swir_dir = split_folder / 'SWIR'
        images_dir = split_folder / 'images'

        if swir_dir.exists():
            # Create a Windows directory junction if 'images' alias doesn't exist
            if not images_dir.exists():
                try:
                    import _winapi
                    _winapi.CreateJunction(str(swir_dir.resolve()), str(images_dir.resolve()))
                    print(f"Created junction: {images_dir} -> {swir_dir}")
                except Exception as e:
                    print(f"Notice: Could not create junction ({e}), using direct SWIR path.")

            target_dir = images_dir if images_dir.exists() else swir_dir
            txt_path = base_dir / f'{split}.txt'
            
            # Generate or refresh txt list
            img_list = sorted(list(target_dir.glob('*.jpg')))
            with open(txt_path, 'w') as f:
                for img_path in img_list:
                    f.write(f"{os.path.abspath(img_path)}\n")
            print(f"Ready: {txt_path} ({len(img_list)} images)")

    # Ensure dataset.yaml is present
    if not yaml_path.exists():
        yaml_content = f"""path: {base_dir.resolve().as_posix()}
train: train.txt
val: val.txt
test: test.txt

nc: 1
names: ['ship']
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created dataset configuration: {yaml_path}")


def train(args):
    # 1. Dataset verification
    base_dir = Path(r"D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project\3.test\yolo26\NSLSR")
    yaml_path = Path(args.data)
    ensure_dataset(base_dir, yaml_path)

    # 2. Check model weights / config
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file '{model_path}' not found!")
        print("Please check the model path or place 'yolo26n.pt' in the working directory.")
        return

    print(f"\n==========================================")
    print(f" Starting YOLO26n Training on NSLSR Dataset ")
    print(f"==========================================")
    print(f" Model       : {model_path}")
    print(f" Data config : {yaml_path}")
    print(f" Epochs      : {args.epochs}")
    print(f" Batch size  : {args.batch}")
    print(f" Image size  : {args.imgsz}")
    print(f" Device      : {args.device}")
    print(f" Workers     : {args.workers}")
    print(f" Project/Name: {args.project}/{args.name}")
    print(f"==========================================\n")

    # 3. Initialize YOLO model
    model = YOLO(model_path)

    # 4. Start training
    proj_dir = str(Path(args.project).resolve()) if args.project else None
    results = model.train(
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

    print("\nTraining completed successfully!")
    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else (Path(args.project) / args.name)
    print(f"Model weights and metrics saved to: {save_dir / 'weights' / 'best.pt'}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26n on NSLSR SWIR Dataset")
    parser.add_argument('--model', type=str, default='yolo26n.pt', help='Initial weights or model config path (default: yolo26n.pt)')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='Path to dataset.yaml (default: dataset.yaml)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--device', default='0', help="CUDA device, e.g. 0 or 'cpu' (default: 0)")
    parser.add_argument('--workers', type=int, default=4, help='DataLoader worker threads (default: 4)')
    parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer (default: auto)')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate (default: 0.01)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience epochs (default: 50)')
    parser.add_argument('--project', type=str, default='runs/train', help='Save directory project name (default: runs/train)')
    parser.add_argument('--name', type=str, default='yolo26n_nslsr', help='Experiment save name (default: yolo26n_nslsr)')
    parser.add_argument('--exist-ok', action='store_true', default=True, help='Existing project/name ok, do not increment')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Whether to use pretrained weights')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from last checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)

# python train.py --model yolo26s.pt --name yolo26s_nslsr 