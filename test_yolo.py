import os
from pathlib import Path
from ultralytics import YOLO

def main():
    base_dir = Path(r"D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project\3.test\yolo26\NSLSR")
    
    # 1. Generate txt files containing image paths for train, val, test
    # YOLO automatically maps '/images/' -> '/labels/' in paths. 
    # We create an 'images' directory junction pointing to 'SWIR' if not already present.
    for split in ['train', 'val', 'test']:
        split_folder = base_dir / f'NSLSR_{split}'
        swir_dir = split_folder / 'SWIR'
        images_dir = split_folder / 'images'
        
        if swir_dir.exists():
            if not images_dir.exists():
                try:
                    import _winapi
                    _winapi.CreateJunction(str(swir_dir.resolve()), str(images_dir.resolve()))
                except Exception:
                    pass
            
            target_dir = images_dir if images_dir.exists() else swir_dir
            txt_path = base_dir / f'{split}.txt'
            img_list = sorted(list(target_dir.glob('*.jpg')))
            with open(txt_path, 'w') as f:
                for img_path in img_list:
                    f.write(f"{os.path.abspath(img_path)}\n")
            print(f"Created {txt_path} with {len(img_list)} images.")

    # 2. Create dataset.yaml
    yaml_path = 'dataset.yaml'
    yaml_content = f"""path: {base_dir.resolve().as_posix()}
train: train.txt
val: val.txt
test: test.txt

# Modify nc (number of classes) and names according to your actual classes
nc: 1
names: ['ship']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created {yaml_path}")

    # 3. Load model and run evaluation
    import sys
    # model_path = sys.argv[1] if len(sys.argv) > 1 else 'yolo26n.pt'
    model_path = r'D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project\3.test\yolo26\runs\train\yolo26s_nslsr_pgd\weights\last.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_path}' not found in the current directory.")
        print(f"Please ensure you have placed '{model_path}' in this folder or pass the model path as an argument:")
        print(f"  python test_yolo.py path/to/your_model.pt")
        return

    print(f"Loading model {model_path}...")
    model = YOLO(model_path)
    
    print("Running validation on NSLSR_test (SWIR)...")
    # Using split='test' will evaluate on the dataset specified by 'test: test.txt' in dataset.yaml
    results = model.val(data=yaml_path, split='test')
    print("Evaluation completed.")

if __name__ == '__main__':
    main()
