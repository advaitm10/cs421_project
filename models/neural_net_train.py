import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_union
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import logging
import datetime
import os
import re
import sys

MAX_NO_IMPROVEMENT = 5

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

if __name__ == "__main__":
    handler = DataHandler('../data/eng_train.csv', '../data/eng_test.csv')

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
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    cur_date_time = re.sub(r'\s|:', '_', str(datetime.datetime.now().replace(microsecond=0)))
    output_dir = f'./nn_ckpt/{cur_date_time}'
    os.mkdir(output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "training.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    num_epochs = 100
    best_test_loss = np.inf
    no_improvement = 0

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)

            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(1), labels.float()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### LOGGING
            if batch_idx % 10 == 0:
                logging.info(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                    f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                    f" | Train Loss: {loss:.2f}\n")

        model.eval()

        # Get metrics on test data
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

            if test_loss < best_test_loss:
                no_improvement = 0
                logging.info(f"Test loss: {test_loss:.2f}\n")

                # Show score summary
                test_preds = (np.array(all_probs) >= 0.5).astype(int)
                logging.info(classification_report(test_preds, all_labels))

                # Save model
                output_path = os.path.join(output_dir, f'model_epoch_{epoch + 1}.pth')

                torch.save(model.state_dict(), output_path)
                best_test_loss = test_loss

            else:
                logging.info(f"Test loss did not decrease: {test_loss:.2f}\n")
                no_improvement += 1

                if no_improvement >= MAX_NO_IMPROVEMENT:
                    logging.info("Reached maximum number of epochs without validation loss improvement. Exiting early.")
                    break