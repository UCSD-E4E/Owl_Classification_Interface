Download the Pre-Trained Model here and save it in '~/Owl_Classification_Interface/Binary_Model' if you do not want to train it. 

https://drive.google.com/file/d/1OPO443x8-EDNRptoSY5KlzK301Lz070i/view?usp=sharing

Results:

![resnet18-result-98 4](https://user-images.githubusercontent.com/49630480/135965211-28a50c75-71e9-4548-8d7f-3de03c2fcb7c.png)

### Usage:
1. Train a new model with 20 epochs and random seed of 100
```bash
python3 torch_train.py --num_epochs 20 --seed 100
```

2. Load and evaluate saved model
```bash
python3 torch_train.py --load ./PyTorch_Binary_Classifier.pth
```