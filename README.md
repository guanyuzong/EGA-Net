# EGA-Net
EGA-Net: Edge Feature Enhancement and Global Information Attention Network for RGB-D Salient Object Detection

## Attention!!!
It is recommended to reproduce our code in the Linux system. 

If you want to run the test.py in Windows, please make sure that the path where you save the test results exists.

## Experimental environment 

python==3.7.0

pytorch==1.8.0

torchvision==0.9.0

tensorboardX==2.5

opencv-python==4.5.5.64

## Training
If you want to retrain our network, we recommend that you follow these steps.

1. Complete the experimental environment setup as described above.

2. Modify the parameters in train.py according to your computer configuration, such as gpu, batchsize.

3. Download the dataset and place it in the data folder [0617] https://pan.baidu.com/s/1L9l1p9WS2T0lh-ZOvkMKJQ].

4. Download the pre-trained model of Resnet-50 and place it in the model_zoo folder [0617] https://pan.baidu.com/s/10uVVY3lvAQ9V0onh0VQlwA.

5. Open terminal. run：python3 train.py

## Testing
If you would like to reproduce our results, please follow these steps.

1. We provide a link to download the parameters of the trained model [0617] https://pan.baidu.com/s/1fhTWP_oplAFpifnPc-O3gQ].

2. Place the parametric model under the './best/modal/' path.

3. Open terminal. run：python3 test.py

4. We also provide links to download the results of our experiments [0617] https://pan.baidu.com/s/1RQ3L--14CQCHox2W6Wp-dQ].

## Evaluation
If you would like to evaluate our entire model parameters through quantitative metrics, please follow these steps.

1. Download the results of our experiments and place them in any path.

2. The evaluation metric code has been placed in the eval_code folder, please use MATLAB to open it.

3. Modify the path to the dataset in main.m.

4. run main.m.
