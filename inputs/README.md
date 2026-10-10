# Inputs Directory Structure

Organize your custom experiment bundles here to run them with the `--input` flag:

```
inputs/
└── <your_experiment_name>/
    ├── teacher/
    │   └── teacher_model.pt     (Full torch.save(model) or state_dict + arch.txt / model.py)
    ├── student/
    │   └── student_model.pt     (Full torch.save(model) or state_dict + arch.txt / model.py)
    └── dataset/
        ├── class_0/             (Standard ImageFolder format)
        ├── class_1/
        └── ...
```

### Running Your Bundle:
```bash
python -m compression_suite.cli sweep --input inputs/<your_experiment_name>
```

### Output Resolution:
Results are saved to `outputs/<your_experiment_name>`. If that directory already exists and contains files, the suite automatically prompts you to create a new version (`outputs/<your_experiment_name>_1`) or overwrite!
