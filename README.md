# DM-UAP

## Dependencies
The project is recommended to be used with ython 3.9 and PyTorch 1.8.1. All dependencies can be installed with following command:
```
pip install -r requirements.txt
```


## Training
To start training, run imagenet_attack.py 
```
python imagenet_attack.py  
  --model VGG19 \
  --dataset ImageNet \
  --batch_size 128 \
  --epochs 60 \
  --lr 0.001 \
  --gamma 0.2 \
  --alpha1 0.05 \
  --alpha2 0.01 \
  --beta1 0.0085 \
  --beta2 0.12 \
  --train_num 10000 \
  --tau 0.1 \
  --allow 10./255
```
This is to craft a UAP from the surrogate model VGG19. More details can be found in [imagenet_attack.py](imagenet_attack.py).

## Testing
To start testing, run test.py 
```
  python test.py
--model vgg19
--noise_path /VGG19/noise.pth
--flow_path /VGG19/flow.pt
```
This will start testing your uap on model VGG19, and record the results in result.log. More details can be found in [test.py](test.py).

## Acknowledgements
