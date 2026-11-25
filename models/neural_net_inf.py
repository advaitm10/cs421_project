import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_union
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 60),
            torch.nn.ReLU(),

            torch.nn.Linear(60, 60),
            torch.nn.ReLU(),

            torch.nn.Linear(60, 60),
            torch.nn.ReLU(),

            torch.nn.Linear(60, 1)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

class DataHandler():
    def __init__(self, train_path, test_path, tf_idf = False, chargram = False, stop_words = None):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if tf_idf:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words, dtype = np.float32)
            if chargram:
                vectorizer = make_union(
                    vectorizer,
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), stop_words=stop_words, dtype = np.float32)
                )
        else:
            vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words, dtype = np.float32)
            if chargram:
                vectorizer = make_union(
                    vectorizer,
                    CountVectorizer(analyzer="char_wb", ngram_range=(2, 5), stop_words=stop_words, dtype = np.float32)
                )

        vectorizer.fit(train_df['text'])

        self.train_data = VecDataset(vectorizer.transform(train_df['text']), train_df['polarization'])
        self.test_data = VecDataset(vectorizer.transform(test_df['text']), test_df['polarization'])


class VecDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.features = X.tocsr()
        self.labels = y

    def __getitem__(self, index):
        x = torch.from_numpy(self.features[index].toarray().squeeze(0)).float()
        return x, torch.tensor(self.labels[index], dtype=torch.long)
    
    def __len__(self):
        return self.labels.shape[0]


if __name__ == '__main__':
    handler = DataHandler('../data/eng_train.csv', '../data/eng_test.csv', stop_words='english')
    train_loader = DataLoader(
        dataset=handler.train_data,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=handler.test_data,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = NeuralNetwork(handler.train_data.features.shape[1])
    state_dict = torch.load("./nn_ckpt/2026-11-24_15_14_30_count_stop_words/model_epoch_1.pth", weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        test_loss = 0
        all_probs = []
        all_labels = []

        for batch_idx, (features, labels) in enumerate(test_loader):
            test_logits = model(features)
            test_probs = torch.sigmoid(test_logits)
            test_loss += F.binary_cross_entropy_with_logits(
                test_logits.squeeze(1), labels.float()
            )

            all_probs.extend(test_probs)
            all_labels.extend(labels)

        # Show score summary
        test_preds = (np.array(all_probs) >= 0.5).astype(int)
        print(classification_report(all_labels, test_preds))
        print(f"Macro Average F1: {f1_score(all_labels, test_preds, average='macro')}")