from typing import Iterable, List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy


class Text2Vecotor:
    def __init__(self,
                 lang: str = "en",
                 vector_type: str = "tfidf",
                 pos: List[str] = ["ADJ", "NOUN", "PRON", "PROPN", "VERB"],
                 stop_words: List[str] = [],
                 max_features: int = 10000):

        self.lang = lang
        self.vector_type = vector_type
        self.vectorizer = None
        self.pos = pos
        self.stop_words = stop_words
        self.max_features = max_features
        self.language_models = {
            "en": "en_core_web_sm"
        }
        self.transformers = {
            "en": "en_core_web_trf"
        }

    def fit(self, texts: Iterable[str]):
        if self.vector_type in ["tfidf", "tf-idf", "tf_idf"]:
            tokens = self._create_tokens(texts)
            self.vectorizer = TfidfVectorizer(
                stop_words=self.stop_words,
                max_features=self.max_features)
            self.vectorizer.fit(tokens)

        elif self.vector_type == "embedding":
            pass

        elif self.vector_type == "transformer":
            pass

        else:
            raise RuntimeError(f"Unavailable vector type {self.vector_type}")

    def transform(self,
                  texts: Iterable[str]) -> Iterable[Iterable[np.float64]]:
        if self.vector_type in ["tfidf", "tf-idf", "tf_idf"]:
            tokens = self._create_tokens(texts)
            return self.vectorizer.transform(tokens).toarray()

        elif self.vector_type == "embedding" or self.vector_type == "transformer":
            if self.vector_type == "transformer":
                model_name = self.transformers[self.lang]
            else:
                model_name = self.language_models[self.lang]

            nlp = spacy.load(model_name)
            disable = ["parser", "ner", "textcat"]
            docs = list(nlp.pipe(texts, disable=disable))

            vectors = [doc.vector for doc in docs]
            return vectors

        else:
            raise RuntimeError(f"Unavailable vector type {self.vector_type}")

    def fit_transform(self,
                      texts: Iterable[str]) -> Iterable[Iterable[np.float64]]:
        self.fit(texts)
        return self.transform(texts)

    def _create_tokens(self, texts: Iterable[str]) -> List[str]:
        nlp = spacy.load(self.language_models[self.lang])
        disable = ["parser", "ner", "textcat"]
        docs = list(nlp.pipe(texts, disable=disable))

        preprocessed_texts = []
        for doc in docs:
            preprocessed_texts.append(
                " ".join([token.lemma_ for token in doc if token.pos_ in self.pos]))

        return preprocessed_texts
