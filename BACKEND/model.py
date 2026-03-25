import torch
import torch.nn as nn
from pathlib import Path
import re
import string
from textblob import TextBlob
import emoji
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk import word_tokenize, sent_tokenize

base_dir = Path(__file__).resolve().parent.parent
vocab = torch.load(f"{base_dir}/MODEL/vocab.pth")





def clean_tweet(text):
  text = re.sub(r'https?://\S+|www\.\S+','',text) # REMOVING URL's
  pattern = re.sub(r'@\w+','',text) # REMOVING ID starts with @ of twitter
  return pattern


def remove_pun(text):
  exclude = string.punctuation
  for char in exclude:
    text = text.replace(char,'')
  return text



chat_word ={
    "A3": "Anytime Anywhere Anyplace",
    "ADIH": "Another Day In Hell",
    "AFK": "Away From Keyboard",
    "AFAIK": "As Far As I Know",
    "ASAP": "As Soon As Possible",
    "ASL": "Age Sex Location",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "BAE": "Before Anyone Else",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRUH": "Bro",
    "BRT": "Be Right There",
    "BSAAW": "Big Smile And A Wink",
    "BTW": "By The Way",
    "BWL": "Bursting With Laughter",
    "CSL": "Cant Stop Laughing",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "DM": "Direct Message",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FIMH": "Forever In My Heart",
    "FOMO": "Fear Of Missing Out",
    "FR": "For Real",
    "FWIW": "For What Its Worth",
    "FYP": "For You Page",
    "FYI": "For Your Information",
    "G9": "Genius",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GMTA": "Great Minds Think Alike",
    "GN": "Good Night",
    "GOAT": "Greatest Of All Time",
    "GR8": "Great",
    "HBD": "Happy Birthday",
    "IC": "I See",
    "ICQ": "I Seek You",
    "IDC": "I Dont Care",
    "IDK": "I Dont Know",
    "IFYP": "I Feel Your Pain",
    "ILU": "I Love You",
    "ILY": "I Love You",
    "IMHO": "In My Honest Opinion",
    "IMU": "I Miss You",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "IYKYK": "If You Know You Know",
    "JK": "Just Kidding",
    "KISS": "Keep It Simple Stupid",
    "L": "Loss",
    "L8R": "Later",
    "LDR": "Long Distance Relationship",
    "LMK": "Let Me Know",
    "LMAO": "Laughing My Ass Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "M8": "Mate",
    "MFW": "My Face When",
    "MID": "Mediocre",
    "MRW": "My Reaction When",
    "MTE": "My Thoughts Exactly",
    "NVM": "Never Mind",
    "NRN": "No Reply Necessary",
    "NPC": "Non Player Character",
    "OIC": "Oh I See",
    "OP": "Overpowered",
    "PITA": "Pain In The Ass",
    "POV": "Point Of View",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My Ass Off",
    "RN": "Right Now",
    "SK8": "Skate",
    "STATS": "Your Sex And Age",
    "SUS": "Suspicious",
    "TBH": "To Be Honest",
    "TFW": "That Feeling When",
    "THX": "Thank You",
    "TIME": "Tears In My Eyes",
    "TLDR": "Too Long Didnt Read",
    "TNTL": "Trying Not To Laugh",
    "TTFN": "Ta Ta For Now",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "W": "Win",
    "W8": "Wait",
    "WB": "Welcome Back",
    "WTF": "What The Fuck",
    "WTG": "Way To Go",
    "WUF": "Where Are You From",
    "WYD": "What You Doing",
    "WYWH": "Wish You Were Here",
    "ZZZ": "Sleeping Bored Tired"
}

def chat_conv(text):
  new_text = []
  for w in text.split():
    if w.upper() in chat_word:
      new_text.append(chat_word[w.upper()])
    else:
      new_text.append(w)
  return " ".join(new_text)

def correction(text):
  return str(TextBlob(text).correct())

def change_emj(text):
  text = emoji.demojize(text)
  return text

# tokenizer = get_tokenizer("basic_english")
def tokenizing(text):
    return word_tokenize(text)

def encode(tokens,vocab):
  encoded = []
  for word in tokens: # tokens = ['i','am','upreti']
    if word in vocab:
      encoded.append(vocab[word])
    else:
      encoded.append(vocab['<UNK>'])
  return encoded

def encoded_wrapper(x):
  return encode(x,vocab)

def pad_sequence(sequences,max_len,pad_value =  0):
  # max_len = max(len(seq) for seq in sequences)
  padded = []

  for seq in sequences:
    seq = seq[:max_len]
    padded_seq = seq + [pad_value]*(max_len - len(seq))
    padded.append(padded_seq)

  return torch.tensor(padded,dtype = torch.long)



def preprocess_text(text):

    text = text.lower()

    text = clean_tweet(text)

    text = change_emj(text)

    text = remove_pun(text)

    text = chat_conv(text)

    tokens = tokenizing(text)

    encoded = encode(tokens, vocab)

    max_length = 30
    padded = pad_sequence([encoded], max_length)

    return padded









class ActionModel(nn.Module):
  def __init__(self,vocab_size,embed_dim,hidden_dim):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size,embed_dim,padding_idx = 0) 
    self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first = True) 
    self.dropout = nn.Dropout(0.3)
    self.fc = nn.Linear(hidden_dim,1)

  def forward(self,x):
    x = self.embeddings(x)
    _,(hidden,_) = self.lstm(x) 
    out = self.fc(self.dropout(hidden[-1])) 
    return out
  


model = ActionModel(len(vocab),128,128)
state_dict = torch.load(f"{base_dir}/MODEL/12.pth",map_location="cpu")
model.load_state_dict(state_dict=state_dict)
model.eval()





def inference(text):
  with torch.no_grad():
    output = model(preprocess_text(text))
    probabilities = torch.sigmoid(output)
    pred_class = (probabilities>0.5).int()
    conf = probabilities
  return pred_class.item(),conf.item()