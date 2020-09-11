from sklearn.model_selection import train_test_split
import numpy as np
import random
from SpacyToolKit.metrics import evaluate

def cross_val_score(model, data, label, n_splits=5, shuffle=False, seed=42):
  data = np.array(data)
  random.seed(seed)
  if shuffle:
    data = random.shuffle(data)
  data = np.split(data, n_splits)

  precision = []
  recall = []
  f1 = []

  for fold in data:
    p, r, f = evaluate(model, fold)[label].values()
    precision.append(p)
    recall.append(r)
    f1.append(f)

  return [(label, np.mean(lst)) for label, lst in (("precision", precision), ("recall", recall), ("f1", f1))]

def train_test_val_split(data, test_size=0.3, val_size=0.3, random_state=42):
  Train, Test = train_test_split(data, train_size=1-(test_size + val_size), random_state=random_state)
  Val, Test = train_test_split(Test, test_size=test_size/(test_size + val_size), random_state=random_state) 
  return Train, Test, Val