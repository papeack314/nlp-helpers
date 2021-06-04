from dataclasses import dataclass
from typing import List
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    data: List[str]
    labels: List[str]


class NewsDataset:
    def __init__(self, categories=None, val_size: float = 0.2, random_seed: int = 0):
        news_train = fetch_20newsgroups(subset="train", categories=categories)
        news_test = fetch_20newsgroups(subset="test", categories=categories)
        train_data, val_data, train_label, val_label = train_test_split(news_train.data,
                                                                        news_train.target,
                                                                        test_size=val_size,
                                                                        random_state=random_seed)

        self.train = Dataset(train_data, train_label)
        self.val = Dataset(val_data, val_label)
        self.test = Dataset(news_test.data, news_train.target)
