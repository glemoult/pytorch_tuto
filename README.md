# Py Torch tutoriel

Quick tuto for Py Torch usage

## Installation

Install dependencies.

```bash
pip install -r requirements.txt
```

## Usage

```python
py torch_tuto.py
```

## Output

```python
Utilisation de : cuda
Files already downloaded and verified
Files already downloaded and verified
CNNModel(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=4096, out_features=1024, bias=True)
  (dropout1): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=1024, out_features=512, bias=True)
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc3): Linear(in_features=512, out_features=10, bias=True)
)
Epoch 1, Loss: 1.5208
Epoch 2, Loss: 1.1134
Epoch 3, Loss: 0.9403
Epoch 4, Loss: 0.8182
Epoch 5, Loss: 0.7184
Epoch 6, Loss: 0.6304
Epoch 7, Loss: 0.5470
Epoch 8, Loss: 0.4866
Epoch 9, Loss: 0.4248
Epoch 10, Loss: 0.3793
Epoch 11, Loss: 0.3453
Epoch 12, Loss: 0.3138
Epoch 13, Loss: 0.2893
Epoch 14, Loss: 0.2684
Epoch 15, Loss: 0.2530
Epoch 16, Loss: 0.2401
Epoch 17, Loss: 0.2314
Epoch 18, Loss: 0.2078
Epoch 19, Loss: 0.2119
Epoch 20, Loss: 0.1992
Entraînement 1 terminé !
Précision 1 du modèle : 72.00%
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)