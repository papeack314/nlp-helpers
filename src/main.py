from corpus import NewsDataset
from preprocessors import Text2Vecotor
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    dataset = NewsDataset()

    t2v = Text2Vecotor(lang="en", vector_type="tf-idf")
    X_train, y_train = t2v.fit_transform(dataset.train.data), dataset.train.labels
    X_val, y_val = t2v.transform(dataset.val.data), dataset.val.labels
    X_test, y_test = t2v.transform(dataset.test.data), dataset.test.labels

    model = CatBoostClassifier(
        iterations=10000,
        early_stopping_rounds=100)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    print(f"Train Accuracy: {accuracy_score(y_train, train_pred)}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred)}")
