# Week 2 Project: Monitoring distribution shift

```
pip install -e .
```

This project will have learners monitor deep learning systems for distribution shift using their predictions.

### Download dataset

In your terminal, please run
```
pip install gdown
```
Then, run the following:
```
gdown --id 1hdwWYFNJiCI_zFHWTMce6TETyj20GHG9
```
This will download a 3gb file of precomputed text embeddings to a file `data.zip`. Unzip the file and replace `monitor_project/data` with the unzipped folder. 


## Baseline Accuracy

**Results on English reviews:**
- Accuracy: 88.26%
- Loss: 0.295

**Results on Spanish reviews:**
- Accuracy: 54.88%
- Loss: 1.046

## Retrained Accuracy:

**Results on English reviews:**
- Accuracy: 88.26%
- Loss: 0.297

**Results on Spanish reviews:**
- Accuracy: 72.99%
- Loss: 0.544