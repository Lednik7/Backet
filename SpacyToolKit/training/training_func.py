import spacy
import random
from tqdm import tqdm
from SpacyToolKit.metrics import evaluate

def create_blank(TRAIN_DATA, lang="en"):
  nlp = spacy.blank(lang)
  ner = nlp.create_pipe("ner")
  nlp.add_pipe(ner, last=True)

  for _, annotations in TRAIN_DATA:
      for ent in annotations.get('entities'):
          ner.add_label(ent[2])
  return nlp, ner
  
def begin_training(nlp, TRAIN_DATA, VAL_DATA=False, n_iter=1, drop_out=0.5, progressbar=True, seed=42):
  random.seed(seed)
  other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
  with nlp.disable_pipes(*other_pipes):
      optimizer = nlp.begin_training()
      for itn in range(n_iter):
          print("\n Iterations:", itn+1)
          random.shuffle(TRAIN_DATA)
          losses = {}
          for text, annotations in tqdm(TRAIN_DATA, disable=not progressbar):
            nlp.update(
                  [text],  # batch of texts
                  [annotations],  # batch of annotations
                  drop=drop_out,  # dropout - make it harder to memorise data
                  sgd=optimizer,  # callable to update weights
                  losses=losses)
          if VAL_DATA:
            print("\n", evaluate(nlp, VAL_DATA))
            
          print(f"\n Losses: {losses}")
  print("Done")
  return nlp