from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import random


class TextDataset(Dataset):
    def __init__(self, root_path, is_training):
        self.is_training = is_training
        if is_training:
            data_path = os.path.join(root_path, "train.csv")
            self.data = pd.read_csv(data_path).reset_index()
            self.non_adult_data = self.data[
                self.data["Category"] != "Adult"
            ].reset_index()
        else:
            data_path = os.path.join(root_path, "test.csv")
            self.data = pd.read_csv(data_path).reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rows = self.data.iloc[index]
        text = rows["Description"]
        label = rows["Category"] == "Adult"
        if label & self.is_training:
            random_threshold = random.random()
            if random_threshold > 0.8:
                random_non_adult_index = random.randint(0, len(self.non_adult_data) - 1)
                random_non_adult_row = self.non_adult_data.iloc[random_non_adult_index]
                text = text + ". " + random_non_adult_row["Description"]
            elif random_threshold > 0.6:
                random_non_adult_index = random.randint(0, len(self.non_adult_data) - 1)
                random_non_adult_row = self.non_adult_data.iloc[random_non_adult_index]
                text = random_non_adult_row["Description"] + ". " + text
        return text, torch.tensor(label).float().unsqueeze(0)


if __name__ == "__main__":
    dataset = TextDataset("dataset", True)
    print(dataset[0])
