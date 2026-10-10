# ResNet-50 -> ResNet-18 Input Bundle

- **Teacher**: ResNet-50 (`teacher/resnet50.pt`)
- **Student**: ResNet-18 (`student/resnet18.pt`)
- **Dataset**: 10 classes (`dataset/`)

### Run Command:
```bash
python -m compression_suite.cli sweep --input ./inputs/resnet_cifar10
```
