import os
import tqdm
import fire
import logging
import string
import re
import csv
import pickle
import math

from scipy.stats import rankdata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

import numpy as np
import pandas as pd


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)


class SSESTM:
    def __init__(self, alpha_plus=0.3, alpha_minus=0.3, kappa=3,
                 reg=0.05, alpha_rate=0.001, max_iters=1000000, error=0.00000001, skip_params_gen=True):
        self.path = ""
        self.skip_params_gen = bool(skip_params_gen)

        # Hyper parameters for init
        self.alpha_plus = float(alpha_plus)
        self.alpha_minus = float(alpha_minus)
        self.kappa = float(kappa)

        # Parameters for data loading
        self.codes = []
        self.returns = []
        self.articles = []
        self.articles_words = []
        self.articles_words_count = []
        self.word_set = []

        # Screened words
        self.S = []
        self.S_count = []

        # P/N Topic parameter for words
        self.O = []

        self.d = []

        # Tuning parameter
        self.reg = float(reg)
        self.alpha_rate = float(alpha_rate)
        self.max_iters = int(max_iters)
        self.error = float(error)

    def _load_data(self, path):
        logging.info("Loading data...")
        codes = []
        returns = []
        articles = []
        df = pd.read_excel(path)

        for idx, row in df.iterrows():
            if idx == 0:
                continue
            try:
                returns.append(float(row["Return2"]) * 100)
            except:
                print(idx)
                continue

            try:
                articles.append(row["Content"])
            except:
                returns.pop()
                print(idx)

        self.codes = codes
        self.returns = returns
        self.articles = articles

    def _preprocess(self):
        logging.info("Preprocessing...")
        articles_words = []
        returns = []
        for idx, article in enumerate(tqdm(self.articles)):
            if not pd.isna(article):
                articles_words.append(self.preprocess_article(article))
                returns.append(self.returns[idx])
        self.articles_words = articles_words
        self.returns = returns
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

        for jdx, article_words in enumerate(tqdm(self.articles_words)):
            total_cnt = 0
            for word in self.S:
                if word in article_words:
                    total_cnt += 1
            self.articles_words_count.append(total_cnt)

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
                if self.articles_words_count[idx] == 0:
                    di.append(0)
                    continue
                di.append(article_words.count(word) / self.articles_words_count[idx])
            d.append(np.array(di, dtype=float))
        d = np.array(d, dtype=float)
        O = np.matmul(d.T.dot(W.T), np.linalg.inv(WWT))
        O[O < 0] = 0
        d = np.sum(d, axis=0)
        self.d = np.array(d, dtype=float)

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

        with open('./data/d.pickle', 'wb') as f:
            pickle.dump(self.d, f, pickle.HIGHEST_PROTOCOL)

    def _load_params(self):
        with open('./data/O.pickle', 'rb') as f:
            self.O = pickle.load(f)

        with open('./data/S.pickle', 'rb') as f:
            self.S = pickle.load(f)

        with open('./data/d.pickle', 'rb') as f:
            self.d = pickle.load(f)

    def _predict(self, article):
        p = np.random.rand(1, 1)[0][0]

        preprocessed_article = self.preprocess_article(article)

        s_hat = 0
        s_words = []
        for word in self.S:
            if word in preprocessed_article:
                s_words.append(word)
                s_hat += 1

        prev_cost = -1

        for itr in range(self.max_iters):
            cost = self._cost(s_words, p)
            gradient = self._gradient(s_words, p)
            p = p - self.alpha_rate * gradient
            # logging.info(f"cost: {cost}, p: {p}, gradient: {gradient}")

            if prev_cost == -1:
                prev_cost = cost
                continue

            if prev_cost - cost < self.error:
                break
            prev_cost = cost

        # logging.info(f"sentiment for this article p: {p}")
        return p

    def _cost(self, s_words, p):
        total_sum = 0
        for idx, word in enumerate(s_words):
            j = self.S.index(word)
            dj = self.d[j]
            total_sum += dj * math.log(p * self.O[j][0] + (1 - p) * self.O[j][1])

        if len(s_words) == 0:
            return - self.reg * math.log(p * (1 - p))

        return - total_sum / len(s_words) - self.reg * math.log(p * (1 - p))

    def _gradient(self, s_words, p):
        total_sum = 0
        for idx, word in enumerate(s_words):
            j = self.S.index(word)
            dj = self.d[j]
            total_sum += dj * (self.O[j][1] - self.O[j][0]) / (self.O[j][0] * p + self.O[j][1] * (1 - p))

        if len(s_words) == 0:
            return self.reg * (2 * p - 1) / (p * (1 - p))

        return total_sum / len(s_words) + self.reg * (2 * p - 1) / (p * (1 - p))

    def preprocess_article(self, article):

        # Remove numbers
        article = re.sub(r'\d+', '', article.lower())

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

        return [word for word in word_tokens if len(word) > 1]

    def train(self, path):
        self._load_data(path)
        self._preprocess()
        self._screen()
        self._calc_topic()
        self._save_params()

    def load_params(self):
        self._load_params()

    def predict(self, article):
        return self._predict(article)

    def run(self):
        logging.info("Running SSESTM")
        if not self.skip_params_gen:
            self._load_data(path="./data/articles.csv")
            self._preprocess()
            self._screen()
            self._calc_topic()
            self._save_params()

        self._load_params()
        f = open("input", "r")
        test_text = f.read()
        self._predict(test_text)


if __name__ == '__main__':
    fire.Fire(SSESTM)
