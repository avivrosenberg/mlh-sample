# Technion-MLH Challenge: Sample Project

This repo contains a sample project to get you started with
the Technion Machine Learning for Healthcare (MLH) Challenge.

## Challenge description

The Technion-MLH Challenge, aims to
stimulate rapid progress on an unsolved question of practical clinical
significance and that may benefit improving healthcare outcome for the citizen
of Israel and globally. For that purpose, data will be provided to the students
together with the medical question of interest.

The dataset consists of a collection of variables and/or physiological signals
that need to be analyzed toward a defined goal. The students are provided with a
subset of the data, the “training set” that will be used by each team to develop
their algorithm. A hidden “test set”, that will not be provided to the students,
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
|---- model.py      # Model and training implementation
|---- run.py        # Training and inference runners
|- data/            # Folder for storing dataset files for training
|- models/          # Folder for saving trained models
|- out/             # Folder for output files
```
