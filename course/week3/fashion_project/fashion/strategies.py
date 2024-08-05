import torch
import numpy as np
from typing import List

from .utils import fix_random_seed
from sklearn.cluster import KMeans

def random_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Randomly pick examples.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  fix_random_seed(42)
  
  indices = []
  # ================================
  # FILL ME OUT
  # Randomly pick a 1000 examples to label. This serves as a baseline.
  # Note that we fixed the random seed above. Please do not edit.
  # HINT: when you randomly sample, do not choose duplicates.
  # HINT: please ensure indices is a list of integers
  # ================================
  # Randomly pick `budget` number of indices without duplicates
  indices = np.random.choice(len(pred_probs), size=budget, replace=False).tolist()

  return indices

def uncertainty_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the model is the least confident in its predictions.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  chance_prob = 1 / 10.  # may be useful
  # ================================
  # FILL ME OUT
  # Sort indices by the predicted probabilities and choose the 1000 examples with 
  # the least confident predictions. Think carefully about what "least confident" means 
  # for a N-way classification problem.
  # Take the first 1000.
  # HINT: please ensure indices is a list of integers
  # ================================
  # Convert torch tensor to numpy array for easier manipulation
  pred_probs_np = pred_probs.numpy()
  
  # Calculate uncertainty as the absolute difference from the uniform probability
  num_classes = pred_probs_np.shape[1]
  chance_prob = 1.0 / num_classes
  uncertainty_scores = np.abs(pred_probs_np - chance_prob).sum(axis=1)
  
  # Find the indices of the `budget` number of most uncertain examples (smallest scores)
  most_uncertain_indices = np.argsort(uncertainty_scores)[:budget]
  
  # Convert to a list of integers
  indices = most_uncertain_indices.tolist()
  return indices

def margin_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the difference between the top two predicted probabilities is the smallest.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  # ================================
  # FILL ME OUT
  # Sort indices by the different in predicted probabilities in the top two classes per example.
  # Take the first 1000.
  # ================================
  # Get the top 2 probabilities for each example
  top_two_probs = torch.topk(pred_probs, 2, dim=1)[0]
  
  # Calculate the margin (difference between the largest and second-largest predicted probability)
  margin = top_two_probs[:, 0] - top_two_probs[:, 1]
  
  # Get the indices of the examples with the smallest margins
  indices = margin.argsort()[:budget].tolist()
  
  return indices

def entropy_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples with the highest entropy in the predicted probabilities.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  epsilon = 1e-6
  # ================================
  # FILL ME OUT
  # Entropy is defined as -E_classes[log p(class | input)] aja the expected log probability
  # over all K classes. See https://en.wikipedia.org/wiki/Entropy_(information_theory).
  # Sort the indices by the entropy of the predicted probabilities from high to low.
  # Take the first 1000.
  # HINT: Add epsilon when taking a log for entropy computation
  # ================================
  # Calculate entropy for each example
  entropy = -torch.sum(pred_probs * torch.log(pred_probs + epsilon), dim=1)
  
  # Sort indices by entropy in descending order (high to low)
  indices = torch.argsort(entropy, descending=True)[:budget]
  
  # Convert to list of integers
  indices = indices.tolist()
  
  return indices
