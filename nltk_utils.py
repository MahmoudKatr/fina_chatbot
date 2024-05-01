import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
        a = "Hello, thanks for visiting"
        print(tokenize(a))
        OUTPUT => ['Hello', ',', 'thanks', 'for', 'visiting']
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
        words = ["ORGanize", "organizes", "organizing"]
        stemmed_words = [stem(w) for w in words]
        print(stemmed_words)
        Output => ['organ', 'organ', 'organ']
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words): # all_words => words
    """
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bog = bag_of_words(sentence, words)
        print(bog)
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag
