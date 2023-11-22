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


## Running

### Training

There is a CLI to train the model. Here is an example of how to run the training. Other options can be found with `poetry run python -m tdt17_project.train --help`: 

```bash
poetry run python -m tdt17_project.train --database-path="path/to/Cityscapes" --epochs=20 --batch-size=8 --learning-rate=0.001 --use-test-set=False --resume-from-weights="path/to/weights.pt"
```

