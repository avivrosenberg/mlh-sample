# Technion-MLH Challenge: Sample Project

This repo contains a sample project to get you started with
the Technion Machine Learning for Healthcare (MLH) Challenge.

## Challenge description

The Technion-MLH Challenge aims to
stimulate rapid progress on an unsolved question of practical clinical
significance and that may benefit improving healthcare outcome for the citizen
of Israel and globally. For that purpose, data will be provided to the students
together with the medical question of interest.

The dataset consists of a collection of variables and/or physiological signals
that need to be analyzed toward a defined goal. The students are provided with a
subset of the data, the training set, that will be used by each team to develop
their algorithm. A hidden test set, that will not be provided to the students,
will be used to analyze the student's algorithm performance.

The data and question we ask are novel and challenge participants will be the
first to research it! 

The Winter 2019 Challenge in partnership with the Rambam Children's hospital
addresses the following research question: 

    For children admitted with bronchiolitis at the Rambam, predict whether the
    length of stay will be prolonged (superior to 2.5 days) by a prior exposure to
    an increased air pollution level. 

In the context of this research, “length of stay” is used as a proxy for the
severity of bronchiolitis and so the underlying question we ask is whether the
air pollution level is an independent variable increasing the severity of
bronchiolitis cases that are admitted at the Rambam. 

## Project structure

```
+
|- environment.yml  # Conda environment file specifying project dependencies
|- mlh.py           # Implements a command line interface for the project
|- mlh_challenge/   # Python package for your submission code
|---- data.py       # Data loading and processing, building features
|---- model.py      # Model implementation
|---- run.py        # Training and inference runners
|---- utils.py      # Reading and writing files, scoring
|- data/            # Folder for storing dataset files for training
|- models/          # Folder for saving trained models
|- out/             # Folder for output files
```

## Working with the code

### Environment set-up

1. Install the python3 version of [miniconda](https://conda.io/miniconda.html).
   Follow the [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   for your platform.
2. Use conda to create a virtual environment for the project.
   From the project root directory, run
   ```shell
   conda env create -f environment.yml
   ```
   This will install all the necessary packages into a new conda virtual
   environment named `mlh-challenge`.
3. Activate the new environment by running
   ```shell
   conda activate mlh-challenge
   ```
   *Activating* an environment simply means that the path to its python binaries
   (and packages) is placed at the beginning of your `$PATH` shell variable.
   Therefore, running programs installed into the conda env (e.g. `python`) will
   run the version from the env since it appears in the `$PATH` before any other
   installed version.
   
   To check what conda environments you have and which is active, run
   ```shell
   conda env list
   ```
   You can find more useful info about conda environments
   [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

Notes: 
- Always make sure the correct environment is active. It will revert to it's
  default each new terminal session.
- On Windows, you can run these commands from the **Anaconda Prompt** program
  that is installed with miniconda. If you also add the `conda` installation
  to the Windows `PATH` variable, you can run these commands from the regular
  windows command prompt.
- If you use PyCharm or any other IDE, you should configure the interpreter path
  of the IDE to the path of the `python` executable within the conda
  env folder. For example, point the interpreter path to
  `~/miniconda3/envs/mlh-challenge/bin/python` if that's where your env is
  installed.
  
### Training

This starter project implements the following training steps:
1. Loading raw data in the format provided by the challenge organizers
2. Processing the data to create features
3. Fitting an example model to the data
4. Saving the fitted model to a file so it can be loaded later
5. Computing the evaluation scores

To perform these steps run e.g.,
```shell script
python mlh.py train --data-file data/mlh-train.npz
```

This will load the training set from the file `data/mlh-train.npz`
and save the model to the default path of `models/model.pkl`
(can be changed using the `--save-model` option).

Notes:
- You can and should edit the implementation of each step as you see fit.
Refer to the `TODO`s in the code as a starting point
for what to work on.
- You can also change parameters or add commands to the CLI,
for example in order to implement a cross-validation runner.
- You can directly run the `training()` function within the `run.py`
module instead of running from the command line.

### Inference

This starter project implements the following steps for inference using an
 **existing pre-trained model**:
1. Loading raw data in the format provided by the challenge organizers
2. Processing the data to create features
3. Loading a pre-trained model from a file
4. Using the loaded model to predict labels of the loaded data
5. Computing the evaluation scores
6. Saving the prediction results to a file

To perform these steps run e.g.,
```shell script
python mlh.py infer --data-file data/mlh-test.npz --load-model models/model.pkl
```
This will load a test set from the file `data/mlh-test.npz`,
load a model from the path `models/model.pkl`, and write an output
file to the default path of `out/results.csv`
(can be changed using the `--out-file` option).

Notes:
- You can change the inference code, as long as you keep the existing
CLI functionality and write the output file in the same way.
We will run this to score your submission.
- You can directly run the `inference()` function within the `run.py`
module instead of running from the command line.

## Submitting

1. When you're done working on your solution and feel confident to submit,
rename the final trained model file to `models/final-model.pkl`.
2. Submit a zip file named `mlh-id1_id2.zip` where `id1` and `id2`
are the IDs of the submitters.
3. The structure inside the zip file should be the same as the project
structure:
    ```shell
    +
    |- mlh.py
    |- mlh_challenge/
    |---- data.py
    |---- model.py
    |---- run.py
    |---- utils.py
    |---- foo.py           # You can add more code files as needed
    |---- bar.py           # You can add more code files as needed
    |- models/
    |---- final-model.pkl  # Your trained model!
    ```

## Evaluation

1. We will evaluate your submission by running inference using the
CLI as shown in the example above.
2. Evaluation will be performed on our hidden test set and using your
final submitted pre-trained model.
3. We will then use the output file `out/results.csv` to compute the ROC-AUC
score.
4. The submissions will be ranked by this score.

