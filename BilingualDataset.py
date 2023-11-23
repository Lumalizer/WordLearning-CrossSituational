import random
from dataclasses import dataclass


@dataclass
class BilingualDataset:
    filename: str = "data/nld.txt"
    percentage: int = 100

    def __post_init__(self):
        with open(self.filename, 'r', encoding="utf-8") as f:
            self.lines = [line.split('\t')[:2] for line in f.readlines()]

        self.lines = [(self.preprocess(line[0]), self.preprocess(line[1])) for line in self.lines]
        self.lines = random.sample(self.lines, int((len(self.lines) // 100)*self.percentage))
        self.lines.sort(key=lambda x: len(x[0].split()))

        random.seed(42)

        self.source_words_total = sum([len(line[0].split()) for line in self.lines])
        self.target_words_total = sum([len(line[1].split()) for line in self.lines])
        self.source_unique_words = set([word for line in self.lines for word in line[0].split()])
        self.target_unique_words = set([word for line in self.lines for word in line[1].split()])

        print(self)

    def shuffle_dataset(self):
        random.shuffle(self.lines)

    def __iter__(self):
        for line in self.lines:
            yield line[0], line[1]

    def __len__(self):
        return len(self.lines)

    def __repr__(self):
        return f"Dataset with {len(self.lines)} sentence pairs, \n{len(self.source_unique_words)} unique source words, {self.source_words_total} total source words, \n{len(self.target_unique_words)} unique target words and {self.target_words_total} total target words."

    def preprocess(self, text: str):
        for char in [",", ".", "?", "!", '"']:
            text = text.replace(char, "")
        return text.replace("'", " ").lower()