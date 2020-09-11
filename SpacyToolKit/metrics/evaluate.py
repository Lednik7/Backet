from SpacyToolKit import spans_to_words

def IoU(model, data):
  scores = []
  for text, annotations in data:
      y_pred = set([(ent.text) for ent in model(text).ents])
      y_true = set(spans_to_words((text, annotations)))
      intersection = len(y_pred.intersection(y_true))
      union = len(y_pred.union(y_true))
      try:
        scores.append(intersection / union)
      except ZeroDivisionError:
         scores.append(0)
  return scores
  
def evaluate(model, examples):
  return model.evaluate(examples).scores['ents_per_type']