# Bayesian Optimization

In this assignment, we will see how to program an instantiation of Sequential Model-based Optimization (SMBO), a specific form of Bayesian Optimization. 

## Outline
SMBO is a simple yet powerful technique, which is described by Hutter et al. (2012) or Snoek et al. (2013). 
SMBO can be written as follows, making use of various auxiliary functions in the data structure. 

```
smbo = SequentialModelBasedOptimization()
smbo.initialize(list(...))

while budget left:
    smbo.fit_model()
    theta_new = smbo.select_configuration(sample_configurations(many))
    performance = optimizee(theta_new)
    smbo.update_runs((theta_new, performance))
```

## Assignment
The assignment consists of three files, 
* `assignment.py`: the file that needs to be filled in order to complete this assignment
* `test.py`: a file with unit tests

In the file `assignment.py`, you are expected to finish the following functions:
* `select_configuration(...)`: Receives an array of configurations, and needs to select the best of these (i.e., the one that most likely yields the most improvement over the current best configuration, &theta;<sub>inc</sub>). It uses an auxiliary function to calculate for each configuration what is the expected improvement. 
* `expected_improvement(...)`: Function to determine for a set of configurations what the expected improvement is. Hint: the `GaussianProcessRegressor` has a predict function that takes as argument `return_std`, to receive both the predicted mean (&mu;) and standard deviation (&sigma;) per configuration. EI is defined in the slides, although note that we require a slightly different formalation, as we are optimizing.
* `update_runs(...)`: Replacement of intensification function. Intensify can only be used when working across multiple random seeds, cross-validation folds, and in this case we have only a single measurement. Intensification will therefore yield no improvement over just running each configuration. This function adds the last run to the list of all seen runs (so the model can be trained on it) and updates &theta;<sub>inc</sub> in case a new best algorithm was found. 
\end{itemize}

Note that for each function that needs to be implemented, you need to write approximately 3-10 lines of code.

## Unit tests

Have a good look at the Unit tests. They specifically demonstrate how Sequential Model-based Optimization works and demonstrate how the interface can be invoked. 
If all unit tests succeed, this is a good indication that the assignment could be completed successfully. 
Hint: make sure to work with the versions of numpy, scipy and scikit-learn as specified in the `requirements.txt` file, to prevent errors due to randomness. 
These are at moment of publication the most recent versions that are compatible with each other. 

## Time limit

All unit tests should not take more than 1 minute combined. 

