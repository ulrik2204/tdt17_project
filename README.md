# Semantic segmentation project for TDT17 Visual Intelligence

## Development
I am using `poetry`, thus initial setup is:
```bash
# Ensure that poetry uses .venv folder
poetry config virtualenvs.in-project true
poetry shell
poetry install
```

## PyTorch
This project uses PyTorch. To install pytorch, install it via pip:
```bash
poetry run python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Dataset
The dataset used is the [Cityscapes dataset](https://www.cityscapes-dataset.com/). The used dataset is located on the IDUN cluster. 



