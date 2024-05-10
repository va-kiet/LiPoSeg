# LiPoSeg
A Lightweight Encoder-Decoder Network for LiDAR-based Road- Object Semantic Segmentation

The implementation code for the paper: [dl.acm.org/doi/abs/10.1145/3628797.3628900](https://dl.acm.org/doi/abs/10.1145/3628797.3628900)

See the paper for more information about the network. 

## Training:
To download the WPI dataset, run the script: `wpi_dataset.m`.

To generate training data from WPI dataset, run the script: `generate_training_data.m`

To start training LiPoSeg, run the script: `train.py`

After training process is done, run `test.m` to see the result.

