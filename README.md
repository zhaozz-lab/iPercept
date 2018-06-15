
# DenseBag: a Neural Network Architecture for Eye Gaze Estimation
---------------------------
GitHub URL: https://github.com/mbbuehler/iPercept


TODO: Intro


# Setup


1. Download [densebag_code.zip](http://mbuehler.ch/public_downloads/densebag/densebag_code.zip) and extract
```shell
	wget http://mbuehler.ch/public_downloads/densebag/densebag_code.zip
	unzip densebag_code.zip
	cd densebag_code
```
2. Make sure you have installed the required python modules.
```python
	python setup.py
```
3. Create a folder and add the [dataset](http://mbuehler.ch/public_downloads/densebag/MPIIGaze_kaggle_students.h5):
```shell
	mkdir datasets 
	cd datasets
	wget http://mbuehler.ch/public_downloads/densebag/MPIIGaze_kaggle_students.h5
```
4. Create the folder for the outputs 
```shell
	mkdir outputs/
```
5. Now you can run the training script from the source folder
```shell
	 cd src/
	 python train_densebag.py -B_start 10 -B 13
```
The next section describes in more detail how to train and make predictions using DenseBag.

# How to Use DenseBag / Reproduce Results

Training the DenseBag model consists of two steps. In the following they are explained in detail.

## Step 1: Bootstrap & Training: Base Models
In order to retrain the DenseBag model, you start by training a number B of base models (DenseNetFixed) using the script `train_densebag.py`. For example, you can train 10 models use `train_densebag.py -B 10`.

The parameter B is set as random seed. Make sure you use different random seeds for training the model. For example if you have already trained 10 models with random seed B=0,...,9, you can train more models using the optional parameter B_start, i.e. `train_densebag.py -B_start 10 -B 15`. This will train another 5 models and use the random seed 10,11,...,14.

The weights of the trained base models are saved in the outputs folder: `../outputs/DenseBag_RS<B>_<timestamp>`. This folder also contains a csv file `to_submit_to_kaggle_<timestamp>.csv` This file contains the predictions for this base model on the test set. In the second step we will average these predictions.

We recommend to train several models in parallel on different machines / GPUs. Again please make sure to use different random seeds, otherwise the final model might not perform optimally.

Note: We trained the model on the [Azure Ubuntu Data Science Machine](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu?tab=Overview) (1 Nvidia Tesla K80 GPU). Training a single model took between 30 and 40 minutes. 

## Step 2: Aggregation: Averaging Predictions

After training the base models you can average their predictions using the script utils/densebag_bag_predictions.py. For a clean file hierarchy we recommend storing the trained models and predictions in a new folder. Here is how to do it:
1. Create a new folder `outputs/DenseBag`
2. Move / copy all output folders for your models (e.g. `DenseBag_RS001_12345` to the newly created folder `outputs/DenseBag`.
3. Go to directory `src/util` and run `densebag_bag_predictions.py`. This will produce a kaggle submission file in the `outputs/DenseBag` folder, e.g. `to_submit_to_kaggle_B_98_1529048062.csv`.

### Possible Issues

We noted that when training several models using train_densebag.py the Azure Virtual Machine freezes after training 5 models in a row. We have not solved this problem (we suspect issues with the VM or a problem with tensorflow). To mitigate this you can train models in batches of 5. If you find out how to fix this issue, please let us know. Thank you.

# Download Pre-Trained Base Models
You might not want to retrain the model. We provide you with a collection of 30 pre-trained base models by [here.](http://mbuehler.ch/public_downloads/DenseBag_trained-models.zip). These models were trained on both the MPII training and validation set using bootstrapping.

# Archive
We experimented with a number of different architectures. The code for building and training these models has been moved to archive folders (e.g. src/archive, src/models/archive,...).

# Acknowledgement
We would like to thank the [ETH AIT Lab](https://ait.ethz.ch/) for organising this challenge and writing the skeleton code. A big thanks goes to [Microsoft Azure](https://azure.microsoft.com/en-gb/) for providing us with the ressources to train DenseBag. Lastly, thanks to [Yixuan Li](https://github.com/YixuanLi/densenet-tensorflow) for the implementation of the [DenseNet Model](http://arxiv.org/abs/1608.06993), which we adapted to our needs.

