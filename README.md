# HST-UAP

## Dependencies
All dependencies can be installed with following command:
```
conda env create -f HST.yml
```

## Baseline Weights
For fair comparison, we provide reproduced baseline weights from prior UAP methods (e.g., SPGD, SGA, DM-UAP) in the baseline/ directory.
These weights are organized as follows:
```bash
baseline/
├── spgd/
│   └── spgd/
│       ├── AlexNet/
│       │   └── spgd_10000_20epoch_125batch.pth
│       ├── VGG19/
│       │   └── ...
│       └── ...
├── sga/
│   └── ...
└── dm-uap/
    └── ...
```
You can use these pre-computed weights for evaluation or comparison without re-training the baselines.
The weights follow the naming convention: {method}_{num_images}_{epochs}_{batch_size}.pth (e.g., spgd_10000_20epoch_125batch.pth).



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
