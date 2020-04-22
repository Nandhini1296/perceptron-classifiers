# perceptron-classifiers
Implementation of vanilla and averaged perceptron classifiers for classification of hotel reviews as either truthful or deceptive, and either positive or negative.

## Data

The training and development contains the following files:

  - A top-level directory with two sub-directories, one for positive reviews and another for negative reviews.
  - Each of the subdirectories contains two sub-directories, one with truthful reviews and one with deceptive reviews.
  - Each of these subdirectories contains four subdirectories, called “folds”.
  - Each of the folds contains 80 text files with English text (one review per file).

## Programs
 
perceplearn.py will learn perceptron models (vanilla and averaged) from the training data, and percepclassify.py will use the models to classify new data.

The learning program will be executed in the following way: ```python perceplearn.py /path/to/input```

The argument is the directory of the training data; the program will learn perceptron models, and write the model parameters to two files: vanillamodel.txt for the vanilla perceptron, and averagedmodel.txt for the averaged perceptron.

The classification program will be invoked in the following way: ```python percepclassify.py /path/to/model /path/to/input```

The first argument is the path to the model file (vanillamodel.txt or averagedmodel.txt), and the second argument is the path to the directory of the test data file; the program will read the parameters of a perceptron model from the model file, classify each entry in the test data, and write the results to a text file called percepoutput.txt in the following format:
``` label_a label_b path1``` \
In the above format, label_a is either “truthful” or “deceptive”, label_b is either “positive” or “negative”, and pathn is the path of the text file being classified.

## Results

### Results for Vanilla Perceptron

Mean F1: 0.8546

### Results for Average Perceptron
Mean F1: 0.8516
