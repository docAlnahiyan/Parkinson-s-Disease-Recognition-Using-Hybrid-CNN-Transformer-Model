# Parkinson's Disease Recognition Using a Hybrid CNN-Transformer Model

Code for the paper:

> A.-N. Khushbu, Z. Yang, Y. Liu, and X. Zhang, "Parkinson's Disease Recognition Using Hybrid CNN-Transformer Model," in *2024 4th International Conference on Industrial Automation, Robotics and Control Engineering (IARCE)*, IEEE, 2024, pp. 307–311. doi: [10.1109/IARCE64300.2024.00064](https://doi.org/10.1109/IARCE64300.2024.00064)
>
> **Best Paper Award**, IARCE 2024.

The paper is available from IEEE Xplore via the DOI above. More about this work: [docalnahiyan.github.io](https://docalnahiyan.github.io/research/).

## What it does

Classifies structural T1-weighted brain MRI into three groups (**Control**, **PD**, **Prodromal**). A ResNet18 extracts local features from the middle axial slice of each scan; a Transformer encoder then models long-range dependencies between them before classification.

On the PPMI dataset (80/20 split) it reaches **80.51% accuracy** (precision 80.49%, recall 79.89%, F1 0.80, AUC-ROC 0.84), ahead of 3D VGG16 (78.36%) and 3D DenseNet121 (77.55%).

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Loads NIfTI scans and takes the middle axial slice; 80/20 split |
| `model.py` | The hybrid CNN-Transformer |
| `train.py` | Training loop, early stopping, metrics and plots |
| `main.py` | Configuration and training entry point; saves `cnn_transformer_pd.pth` |
| `test.py` | Evaluates a saved model |

## Data

The MRI data come from the [Parkinson's Progression Markers Initiative (PPMI)](https://www.ppmi-info.org/), which requires an approved data-use agreement; no scans are included here. Arrange the registered `.nii` scans as:

```
control/   pd/   prodromal/
```

## Requirements

Python 3 with PyTorch, torchvision, nibabel, scikit-learn, NumPy, Matplotlib and seaborn.

## Citation

```bibtex
@inproceedings{khushbu2024parkinsons,
  title     = {{P}arkinson's Disease Recognition Using Hybrid {CNN-Transformer} Model},
  author    = {Khushbu, Al-Nahiyan and Yang, Z. and Liu, Y. and Zhang, X.},
  booktitle = {2024 4th International Conference on Industrial Automation, Robotics and Control Engineering (IARCE)},
  pages     = {307--311},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/IARCE64300.2024.00064}
}
```

## Licence

Code: [MIT](LICENSE). The paper itself is © 2024 IEEE and is not included.
