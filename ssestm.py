import os
import tqdm
import fire
import logging
import string
import re
import csv
import pickle

from scipy.stats import rankdata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

import numpy as np


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)


class SSESTM:
    def __init__(self, path="./data", alpha_plus=0.2, alpha_minus=0.2, kappa=3, lambd=0.1, skip_params_gen=True):
        self.path = path
        self.skip_params_gen = skip_params_gen

        # Hyper parameters for init
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.kappa = kappa

        # Parameters for data loading
        self.codes = []
        self.returns = []
        self.articles = []
        self.articles_words = []
        self.word_set = []

        # Screened words
        self.S = []
        self.S_count = []

        # P/N Topic parameter for words
        self.O = []

        # Tuning parameter
        self.lambd = lambd

    def _load_data(self):
        logging.info("Loading data...")
        codes = []
        returns = []
        articles = []
        with open(os.path.join(self.path, "data.csv"), "r") as fin:
            rdr = csv.reader(fin)
            for idx, line in tqdm(enumerate(rdr)):
                if idx == 0:
                    continue
                if idx == 200:
                    break
                codes.append(line[0])
                returns.append(float(line[1]))
                articles.append(line[2])
        self.codes = codes
        self.returns = returns
        self.articles = articles

    def _preprocess(self):
        logging.info("Preprocessing...")
        articles_words = []
        for article in tqdm(self.articles):
            # Lowercase 1 words
            article = article.lower()

            # Remove numbers
            article = re.sub(r'\d+', '', article)

            # Remove punctuations, symbols
            article = article.translate(str.maketrans('', '', string.punctuation))
            article = " ".join(article.split())

            # Remove stopwords, symbols
            stop_words = set(stopwords.words("english"))
            additional_symbols = ["—", "”", "’", "“", "″"]
            word_tokens = word_tokenize(article)
            word_tokens = [word for word in word_tokens if word not in stop_words and word not in additional_symbols]

            # Lemmatize word tokens
            lemmatizer = WordNetLemmatizer()
            word_tokens = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]

            # Stem word tokens
            stemmer = PorterStemmer()
            word_tokens = [stemmer.stem(word) for word in word_tokens]

            word_tokens = [word for word in word_tokens if len(word) > 1]
            articles_words.append(word_tokens)
        self.articles_words = articles_words
        word_set = []
        for article_words in articles_words:
            for word in article_words:
                if word not in word_set:
                    word_set.append(word)
        self.word_set = word_set

    def _screen(self):
        logging.info("Screening positive/negative words...")
        for word in tqdm(self.word_set):
            total_cnt = 0
            positive_cnt = 0
            for jdx, article_words in enumerate(self.articles_words):
                if word in article_words:
                    total_cnt += 1
                    if self.returns[jdx] > 0:
                        positive_cnt += 1
            if total_cnt > self.kappa:
                fj = positive_cnt / total_cnt
                if fj >= 1/2 + self.alpha_plus:
                    # Positive sentiment terms
                    self.S.append(word)
                    self.S_count.append(total_cnt)
                elif fj <= 1/2 - self.alpha_minus:
                    # Negative sentiment terms
                    self.S.append(word)
                    self.S_count.append(total_cnt)

    def _calc_topic(self):
        logging.info("Calculating topic matrix...")
        p = []
        n = len(self.articles_words)
        for rank in rankdata(self.returns):
            p.append(rank / n)
        W = np.array([p, [1 - pi for pi in p]], dtype=float)
        WWT = W.dot(W.T)

        d = []
        for idx, article_words in enumerate(self.articles_words):
            di = []
            for word in self.S:
                di.append(article_words.count(word) / self.S_count[idx])
            d.append(np.array(di, dtype=float))
        d = np.array(d, dtype=float)
        O = np.matmul(d.T.dot(W.T), np.linalg.inv(WWT))
        O[O < 0] = 0

        O_renormalized = []
        for row in O:
            l1_norm = np.sum(row)
            O_renormalized.append(row / l1_norm)
        self.O = np.array(O_renormalized)

    def _save_params(self):
        logging.info("Saving params...")
        with open('./data/O.pickle', 'wb') as f:
            pickle.dump(self.O, f, pickle.HIGHEST_PROTOCOL)
        with open('./data/S.pickle', 'wb') as f:
            pickle.dump(self.S, f, pickle.HIGHEST_PROTOCOL)

    def _predict(self, article):
        logging.info("Saving params...")
        with open('./data/O.pickle', 'rb') as f:
            data = pickle.load(f)
            print(data)

        with open('./data/S.pickle', 'rb') as f:
            data = pickle.load(f)
            print(data)

    def run(self):
        logging.info("Running SSESTM")
        if not self.skip_params_gen:
            self._load_data()
            self._preprocess()
            self._screen()
            self._calc_topic()
            self._save_params()
        self._predict("")


if __name__ == '__main__':
    fire.Fire(SSESTM)
