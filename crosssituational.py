from collections import defaultdict
from dataclasses import dataclass
import random
from copy import deepcopy
import os
import numpy as np
from word2word import Word2word
import matplotlib.pyplot as plt

@dataclass
class BilingualDataset:
    lines: list[tuple[str, str]]

    def __iter__(self):
        for line in self.lines:
            yield line[0], line[1]

    def __len__(self):
        return len(self.lines)


class CrossSituationalModel:
    @staticmethod
    def get_datasets(filename: str, shuffled: bool = False):
        with open(filename, 'r', encoding="utf-8") as f:
            lines = [line.split('\t')[:2] for line in f.readlines()]

        random.seed(42)
        test_indices = random.sample(range(len(lines)), int(len(lines)))
        cutoff = int(len(lines) * 0.01)
        test_indices = test_indices[:cutoff]

        train_lines = [lines[i] for i in range(len(lines)) if i not in test_indices]
        test_lines = [lines[i] for i in test_indices]

        if shuffled:
            random.shuffle(train_lines)

        return BilingualDataset(train_lines), BilingualDataset(test_lines)

    def __init__(self, name, filename: str = "data/nld.txt", shuffled: bool = False):
        self.name = name
        self.en2nl = Word2word("en", "nl")

        self.train_data, self.test_data = self.get_datasets(filename, shuffled)

        self.source_word_counts = defaultdict(int)
        self.target_word_counts = defaultdict(int)

        self.source_unique_words = set()
        self.target_unique_words = set()
        
        self.table = defaultdict(lambda: defaultdict(int))
        self.translations = defaultdict(str)

        if os.path.exists("data/nld_dict.txt"):
            self.load_translations()

    def load_translations(self):
        with open("data/nld_dict.txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                extra_info = line.find(" [")

                if extra_info != -1:
                    line = line[:line.find(" [")]

                source_word, target_word = line.split()
                self.translations[source_word] = target_word

    def preprocess(self, text: str):
        for char in [",", ".", "?", "!", '"']:
            text = text.replace(char, "")
        return text.replace("'", " ").lower()
    
    def process_sentence_pair(self, pair: tuple[str, str]):
        source_sentence, target_sentence = pair
        source_words = self.preprocess(source_sentence).split()
        target_words = self.preprocess(target_sentence).split()

        for source_word in source_words:
            for target_word in target_words:
                self.table[source_word][target_word] += 1
                self.source_word_counts[source_word] += 1
                self.target_word_counts[target_word] += 1

        self.source_unique_words.update(source_words)
        self.target_unique_words.update(target_words)

    def get_most_likely_translation_old(self, source_word: str):
        target_words = list(self.table[source_word].keys())
        target_words_score = [self.table[source_word][word] / self.target_word_counts[word] for word in target_words]
        target_words_confidence = [self.table[source_word][word] for word in target_words]
        max_index = target_words_score.index(max(target_words_score))
        return target_words[max_index], target_words_confidence[max_index] - 1

    def get_most_likely_translation_bayes(self, source_word: str):
        target_words = list(self.table[source_word].keys())
        smooth = 1

        priors = [(self.table[source_word][word] + smooth) / (self.target_word_counts[word] + smooth * len(target_words)) for word in target_words]
        likelihoods = [(self.table[source_word][word] + smooth) / (self.target_word_counts[word] + smooth * len(target_words)) for word in target_words]

        # normalize
        likelihoods = [likelihood / sum(likelihoods) for likelihood in likelihoods]
        priors = [prior / sum(priors) for prior in priors]

        posteriors = [(likelihood * prior) / sum(np.array(likelihoods) * np.array(priors)) for likelihood, prior in zip(likelihoods, priors)]
        max_index = posteriors.index(max(posteriors))
        return target_words[max_index], posteriors[max_index]

    def train(self):
        results = []

        for i, pair in enumerate(self.train_data):
            self.process_sentence_pair(pair)

            if i % (len(self.train_data) // 100) == 0:
                test_acc = self.test()
                results.append((i, test_acc))
                print(f"Test performance (i={i}): {test_acc}") 

        self.test_results = results
        return results

    def test(self):
        self.prepare_enter_test()

        test_unique_source_words = set()

        for pair in self.test_data:
            source_words = self.preprocess(pair[0]).split()
            for source_word in source_words:
                test_unique_source_words.add(source_word)

            self.process_sentence_pair(pair)

        translation_pairs = [(source_word, *self.get_most_likely_translation_bayes(source_word)) for source_word in test_unique_source_words]
        results = self.evaluate_accuracy(translation_pairs)

        self.cleanup_after_test()
        return results

    def prepare_enter_test(self):
        self.source_word_counts_ = deepcopy(self.source_word_counts)
        self.target_word_counts_ = deepcopy(self.target_word_counts)
        self.table_ = deepcopy(self.table)
        self.source_unique_words_ = deepcopy(self.source_unique_words)
        self.target_unique_words_ = deepcopy(self.target_unique_words)

    def cleanup_after_test(self):
        self.source_word_counts = self.source_word_counts_
        self.target_word_counts = self.target_word_counts_
        self.table = self.table_
        self.source_unique_words = self.source_unique_words_
        self.target_unique_words = self.target_unique_words_

    def evaluate_accuracy(self, translation_pairs: list[tuple[str, str, int]]):
        correct = 0
        missing = []

        for source_word, target_word, confidence in translation_pairs:
            try:
                translations = self.en2nl(source_word)
                if target_word in translations:
                    correct += 1
            except KeyError:
                missing.append((source_word, target_word))

        if missing:
            missing = sorted(missing, key=lambda x: x[0])
            print(f"Missing {len(missing)} words from the test set (out of {len(translation_pairs)} words) written to file.")
            with open("data/missing.txt", "w", encoding="utf-8") as f:
                for source_word, target_word in missing:
                    sentences = [pair for pair in self.test_data if source_word in [self.preprocess(w) for w in pair[0].split()]]
                    f.write(f"{source_word} {target_word} {str(sentences)}\n")

        if len(translation_pairs) == len(missing):
            return 0

        return correct / (len(translation_pairs) - len(missing))
    
    def plot_results(self):
        x, y = zip(*self.test_results)
        plt.plot(x, y)
        plt.title(f"Test performance over time ({self.name})")
        plt.xlabel("Number of training examples")
        plt.ylabel("Test performance")
        plt.show()

    
    def show_translations(self):
        translation_pairs = [(source_word, *self.get_most_likely_translation_bayes(source_word)) for source_word in self.source_unique_words]
        translation_pairs.sort(key=lambda x: x[2], reverse=False)

        for source_word, target_word, confidence in translation_pairs:
            print(f"[ {source_word} ► {target_word} ] (confidence: {confidence})")
    


model_p = CrossSituationalModel("Progressive Input Complexity")
model_p.train()
model_p.plot_results()

model_r = CrossSituationalModel("Random Input Complexity", shuffled=True)
model_r.train()
model_r.plot_results()