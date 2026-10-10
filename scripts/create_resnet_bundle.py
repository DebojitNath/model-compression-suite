"""
Script to create a complete, ready-to-run ResNet input bundle in inputs/.

Creates:
  inputs/<bundle_name>/
    ├── teacher/
    │   └── resnet50.pt
    ├── student/
    │   └── resnet18.pt
    └── dataset/
        ├── airplane/
        ├── automobile/
        └── ... (CIFAR-10 classes)
"""
import argparse
import os
import sys

# Ensure repository root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torchvision
from torchvision.datasets import CIFAR10

from compression_suite.models.registry import build_model


def create_resnet_bundle(
    bundle_dir: str = "./inputs/resnet_cifar10",
    num_classes: int = 10,
    samples_per_class: int = 20,
    download_cifar: bool = True,
):
    print(f"\n=======================================================")
    print(f"Creating ResNet Input Bundle at: {bundle_dir}")
    print(f"=======================================================")

    teacher_dir = os.path.join(bundle_dir, "teacher")
    student_dir = os.path.join(bundle_dir, "student")
    dataset_dir = os.path.join(bundle_dir, "dataset")

    os.makedirs(teacher_dir, exist_ok=True)
    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # 1. Instantiate and Export Teacher ResNet-50
    print(">>> Exporting Teacher (ResNet-50) to teacher/resnet50.pt...")
    teacher = build_model("resnet50", num_classes=num_classes)
    teacher_path = os.path.join(teacher_dir, "resnet50.pt")
    torch.save(teacher, teacher_path)
    print(f"    [+] Saved: {teacher_path} ({os.path.getsize(teacher_path) / (1024*1024):.2f} MB)")

    # 2. Instantiate and Export Student ResNet-18
    print(">>> Exporting Student (ResNet-18) to student/resnet18.pt...")
    student = build_model("resnet18", num_classes=num_classes)
    student_path = os.path.join(student_dir, "resnet18.pt")
    torch.save(student, student_path)
    print(f"    [+] Saved: {student_path} ({os.path.getsize(student_path) / (1024*1024):.2f} MB)")

    # 3. Export Sample Dataset Images into dataset/ class subfolders
    print(f">>> Packaging dataset images into dataset/ ({samples_per_class} per class)...")
    cifar_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ][:num_classes]

    for cname in cifar_classes:
        os.makedirs(os.path.join(dataset_dir, cname), exist_ok=True)

    try:
        # Load local or download CIFAR-10
        raw_cifar = CIFAR10(root="./data", train=True, download=download_cifar)
        class_counts = {i: 0 for i in range(num_classes)}
        saved_total = 0

        for img, label in raw_cifar:
            if label < num_classes and class_counts[label] < samples_per_class:
                cname = cifar_classes[label]
                idx = class_counts[label]
                img_path = os.path.join(dataset_dir, cname, f"sample_{idx}.png")
                img.save(img_path)
                class_counts[label] += 1
                saved_total += 1
                if all(cnt >= samples_per_class for cnt in class_counts.values()):
                    break

        print(f"    [+] Exported {saved_total} images across {num_classes} classes to {dataset_dir}")
    except Exception as e:
        print(f"    [!] Notice: Could not extract CIFAR-10 images directly ({e}). Creating fallback demo images...")
        from PIL import Image
        for l_idx, cname in enumerate(cifar_classes):
            for i in range(samples_per_class):
                dummy = Image.new("RGB", (32, 32), color=(l_idx * 25, i * 10, 150))
                dummy.save(os.path.join(dataset_dir, cname, f"sample_{i}.png"))
        print(f"    [+] Created fallback image structure in {dataset_dir}")

    # Write a bundle README
    readme_path = os.path.join(bundle_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# ResNet-50 -> ResNet-18 Input Bundle\n\n")
        f.write(f"- **Teacher**: ResNet-50 (`teacher/resnet50.pt`)\n")
        f.write(f"- **Student**: ResNet-18 (`student/resnet18.pt`)\n")
        f.write(f"- **Dataset**: {num_classes} classes (`dataset/`)\n\n")
        f.write(f"### Run Command:\n```bash\npython -m compression_suite.cli sweep --input {bundle_dir}\n```\n")

    print(f"\n[SUCCESS] Input bundle created successfully at: {bundle_dir}")
    print(f"You can now run it immediately with:")
    print(f"  python -m compression_suite.cli sweep --input {bundle_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ready-to-run ResNet input bundle inside inputs/")
    parser.add_argument("--output-dir", type=str, default="./inputs/resnet_cifar10", help="Bundle destination directory")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of target classes")
    parser.add_argument("--samples-per-class", type=int, default=25, help="Number of sample images per class")
    args = parser.parse_args()

    create_resnet_bundle(
        bundle_dir=args.output_dir,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
    )
