# EGA-Net
EGA-Net: Edge Feature Enhancement and Global Information Attention Network for RGB-D Salient Object Detection
## Experimental environment 
pytorch==1.8.0

torchvision==0.9.0

tensorboardX==2.5

opencv-python==4.5.5.64

## Training
If you want to retrain our network, we recommend that you follow these steps.

1. Complete the experimental environment setup as described above.

2. Modify the parameters in train.py according to your computer configuration, such as gpu, batchsize.

3. Download the dataset and place it in the data folder.

4. Download the pre-trained model of Resnet-50 and place it in the model_zoo folder.

5. Open terminal. run：python3 train.py

## Testing
If you would like to reproduce our results, please follow the steps below.

1. We provide a link to download the parameters of the trained model.

2. Place the parametric model under the 'best/modal/' path.

3. Open terminal. run：python3 test.py

4. We also provide links to download the results of our experiments.

## Testing
