# Smart Mobility Challenge

## Introduction

This package presents the work of the **Adorable Interns** team in the **Smarter Mobility Data Challenge for a Greener Future** organized by the network **Manifeste IA**. The goal of the challenge was to make forecasts of the states of 91 electric vehicle charging stations in Paris. Our method is detailed in the *Method.pdf* file. 

## Installation

Prerequisites:
Python 3.8

Use pip to install all the required libraries listed in the requirements.txt file.

```bash
pip install -r requirements.txt
```

## Usage

There are two ways to use the package.

1. Using the main function to produce the forecast of the model $C_{exp}(5, 200)$ in the output file.

```bash
python3 main.py
```
The main trains 12 catboost models which takes approximately 15 minutes.

2. Using the step-by-step Jupyer notebook **notebook.ipynb** which complements our paper *Method.pdf*.

Once it finishes running, an **output** folder is created containing the predictions made on the test dataset.

## Resources

The following resources are available:

- Paper: *Method.pdf*
- Github: https://github.com/NathanDoumeche/Smart_mobility_challenge
- Gitlab: https://gitlab.com/alexis.thomasjutisz/Smart_mobility_challenge
- Challenge website: https://codalab.lisn.upsaclay.fr/competitions/7192
