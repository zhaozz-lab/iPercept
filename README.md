# How to Reproduce Results

We trained the model on the Azure Ubuntu Data Science Machine. Training a single model took between 30 and 40 minutes.

In order to retrain the DenseBag model, train B DenseNet models using the script "train_densebag.py". For example, for training 10 models use "train_densebag.py -B 10".

The parameter B is set as random seed. Make sure you use different random seeds for training the model. For example if you have already trained 10 models, you can keep training using the optional parameter B_start, i.e. "train_densebag.py -B_start 10 -B 15". This will train another 5 models and use the random seed 10,11,...

Of course you can train models in parallel on different machines. Again please make sure to use different random seeds.

After training your B models you can average their predictions using the script utils/densebag_bag_predictions.py.
1. Create a new folder "outputs/DenseBag"
2. Copy all output folders for your models (e.g. "DenseBag_RS001_12345" to the newly created folder "outputs/DenseBag".
3. Run "utils/densebag_bag_predictions.py". This will produce a kaggle submission file in the "outputs/DenseBag" folder.

Issues

We noted that when training several models using train_densebag.py the machine stops after training 5 models.
To mitigate this we trained our models using bash "screen" and restarted the script after 5 iterations.

works well (don't forget to change the parameters B and B_start in order not to use the same random seed again).
We did not look into this issue.


#Download Trained Models
You can download all our trained models here TODO




# --------------------------

# Eye Gaze Estimation Skeleton Code
Visit [here](https://ait.ethz.ch/teaching/courses/2018-SS-Machine-Perception/) for more information about the Machine Perception course.

All questions should first be directed to [our course Piazza](https://piazza.com/class/jdbpmonr7fa26b) before being sent to my [e-mail address](mailto:spark@inf.ethz.ch).

## Setup

The following two steps will prepare your environment to begin training and evaluating models.

### Downloading necessary datasets

Simply run

```
cd datasets
bash get_datasets.bash
```

### Installing dependencies

Run (with `sudo` appended if necessary),
```
python3 setup.py install
```

Note that this can be done within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). In this case, the sequence of commands would be similar to:
```
    mkvirtualenv -p $(which python3) myenv
    python3 setup.py install
```

when using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).

## Structure

* `datasets/` - all data sources required for training/validation/testing.
* `outputs/` - any output for a model will be placed here, including logs, summaries, checkpoints, and Kaggle submission `.csv` files.
* `src/` - all source code.
    * `core/` - base classes
    * `datasources/` - routines for reading and preprocessing entries for training and testing
    * `models/` - neural network definitions
    * `util/` - utility methods
    * `main.py` - training script

## Creating your own model
### Model definition
To create your own neural network, do the following:
1. Make a copy of `src/models/example.py`. For the purpose of this documentation, let's call the new file `newmodel.py` and the class within `NewModel`.
2. Now edit `src/models/__init__.py` and insert the new model by making it look like:

```
from .example import ExampleNet
from .newmodel import NewModel
__all__ = ('ExampleNet', 'NewModel')
```

3. Lastly, make a copy or edit `src/main.py` such that it imports and uses class `NewModel` instead of `ExampleNet`.

### Training the model
If your training script is called `main.py`, simply `cd` into the `src/` directory and run
```
python3 main.py
```

### Outputs
When your model has completed training, it will perform a full evaluation on the test set. For class `ExampleNet`, this output can be found in the folder `outputs/ExampleNet/` as `to_submit_to_kaggle_XXXXXXXXX.csv`.

Submit this `csv` file to our page on [Kaggle](https://www.kaggle.com/c/mp18-eye-gaze-estimation/submissions).
=======
