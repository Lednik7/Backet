from googletrans import Translator

def get_translate(text):
  """
  get_translate(text)
  For this function to work, you must install the module - googletrans.
  This function takes one argument - string and returns - string.
  Any other argument will result in an error.
  """
  return Translator().translate(text).text