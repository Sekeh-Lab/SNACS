# Weighted Mututal Information-based Neuron Trimming

## Contents
This file contains the results pertaining to the experiments performed for the wMINT paper


## Main table of results
--------------------------------------------
| Model - Dataset | Compression | Accuracy |
|:---------------:|:-----------:|:--------:|
| VGG16 - CIFAR10 |      N/A    |   93.94  |
--------------------------------------------

## Runtime  vs. Group size results table
-----------------------------------------------------
| Group Size  |    MINT   |   EDGE    | EDGE w/ Phi |
|:-----------:|:---------:|:---------:|:-----------:|
|  16 - 16    |  1161.51  | 45.56     |   46.56     |
|  32 - 32    |  6573.51  | 205.25    |   171.06    |
|  64 - 64    | 14517.92  | 839.93    |   832.46    |
| 128 - 128   | 74125.99  | 2439.35   |   3961.95   |
| 256 - 256   | 271880.62 | 9999.00   |   10028.28  |
-----------------------------------------------------

#### Notes
- We compare all the methods on conv 9 layer since it represents a 512 x 512 i/o setup, the most dense in VGG16.
- Here we use simple Phi = wt
- samples per class is set to 200

## Phi types results table
--------------------------------------------------------------
| Phi Type                          | Compression | Accuracy |
|:---------------------------------:|:-----------:|:--------:|
| Baseline                          |      N/A    |   93.94  |
| MINT                              |    83.43    |   93.43  | (Prune: 15.035583, Params: 14712832)
| Constant=1                        |    --.--    |   --.--  |
| Weights                           |    --.--    |   --.--  |
| Weights**2                        |    --.--    |   --.--  |
| Activations                       |    --.--    |   --.--  |
| Activations**2                    |    --.--    |   --.--  |
| Weights Activations               |    --.--    |   --.--  |
| Weights Activations**2            |    --.--    |   --.--  |
| exp(-Weights**2/2)                |    --.--    |   --.--  |
| exp(-Activations**2/2)            |    --.--    |   --.--  |
| exp(-(Weights Activations)**2/2)  |    --.--    |   --.--  |
--------------------------------------------------------------
 
#### Notes
- We compare all the methods on VGG16 - CIFAR10 combination.
- Running Gamma computations for default setup
