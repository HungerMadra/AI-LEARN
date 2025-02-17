import urllib.request
import re

# Url Holds raw text file from github
url = ('https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt')
file_path = 'the-verdict.txt'
urllib.request.urlretrieve(url, file_path)
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
   # print('Total Number of Characters in the Text:', len(raw_text))
   # print('First 100 Characters:', raw_text[:99])

# We use regex to split the text into words and punctuations (Tokens)
preprocessed = re.split(r"([,.:;?_!()'\"]|--|\s)", raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed))
#print (preprocessed[:30])


# Then we convert the tokens into ID's 
# We start by sorting the list of unique tokens in the preprocessing list 
# and sorting them alphabetically to determine the vocabulary size 

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#print('Vocabulary Size:', vocab_size)

# We then create a dictionary that maps each word to a unique integer ID

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    #print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r"([,.:;?_!()'\"]|--|\s)", text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        return ''.join(
            self.int_to_str[i] if re.match(r"^[,.:;?_!()'\"]|--$", self.int_to_str[i]) 
            else f' {self.int_to_str[i]}'  # Adds space only before words
            for i in ids
        ).lstrip()  # Removes leading space
    

#1 Stores the vocabulary as a class attribute for access in the encode and decode methods
#2 Creates an inverse vocabulary that maps token IDs back to the original text tokens
#3 Processes input text into token IDs
#4 Converts token IDs back into text
#5 Removes spaces before the specified punctuation

tokenizer = SimpleTokenizerV1(vocab)
text = "\"It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))














