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
    filename: str = "data/nld.txt"
    shuffled: bool = False

    def __post_init__(self):
        with open(self.filename, 'r', encoding="utf-8") as f:
            self.lines = [line.split('\t')[:2] for line in f.readlines()]

        if self.shuffled:
            random.shuffle(self.lines)

        self.lines = [(self.preprocess(line[0]), self.preprocess(line[1])) for line in self.lines]
        self.source_words_total = sum([len(line[0].split()) for line in self.lines])
        self.target_words_total = sum([len(line[1].split()) for line in self.lines])
        self.source_unique_words = set([word for line in self.lines for word in line[0].split()])
        self.target_unique_words = set([word for line in self.lines for word in line[1].split()])

        print(self)

    def __iter__(self):
        for line in self.lines:
            yield line[0], line[1]

    def __len__(self):
        return len(self.lines)
    
    def __repr__(self):
        # show some statistics about the dataset
        return f"Dataset with {len(self.lines)} sentence pairs, \n{len(self.source_unique_words)} unique source words, {self.source_words_total} total source words, \n{len(self.target_unique_words)} unique target words and {self.target_words_total} total target words."
    
    def preprocess(self, text: str):
        for char in [",", ".", "?", "!", '"']:
            text = text.replace(char, "")
        return text.replace("'", " ").lower()


class CrossSituationalModel:
    def __init__(self, name, data: BilingualDataset):
        self.name = name
        self.en2nl = Word2word("en", "nl")

        self.data = data

        self.source_word_counts = defaultdict(int)
        self.target_word_counts = defaultdict(int)
        
        self.table = defaultdict(lambda: defaultdict(int))

        self.translation_cache = {}
    
    def process_sentence_pair(self, pair: tuple[str, str]):
        source_sentence, target_sentence = pair
        source_words = source_sentence.split()
        target_words = target_sentence.split()

        for source_word in source_words:
            for target_word in target_words:
                self.table[source_word][target_word] += 1
                self.source_word_counts[source_word] += 1
                self.target_word_counts[target_word] += 1

        return len(source_words)

    def get_most_likely_translation_old(self, source_word: str):
        target_words = list(self.table[source_word].keys())
        target_words_score = [self.table[source_word][word] / self.target_word_counts[word] for word in target_words]
        target_words_confidence = [self.table[source_word][word] for word in target_words]
        max_index = target_words_score.index(max(target_words_score))
        return target_words[max_index], target_words_confidence[max_index] - 1

    def get_most_likely_translation_bayes(self, source_word: str):
        target_words = list(self.table[source_word].keys())

        if not target_words:
            return "", 0

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
        results = [(0, 0)]
        words_processed = 0
        update_interval = self.data.source_words_total // 10

        for i, pair in enumerate(self.data):
            words_processed += self.process_sentence_pair(pair)

            if words_processed >= update_interval or i == len(self.data) - 1:
                test_acc = self.test()
                results.append((i, test_acc))
                print(f"Test performance (i={i}): {test_acc}")
                words_processed -= update_interval

        self.test_results = results
        return results

    def test(self):
        translation_pairs = [(source_word, *self.get_most_likely_translation_bayes(source_word)) for source_word in self.data.source_unique_words]
        results = self.evaluate_accuracy(translation_pairs)
        return results

    def evaluate_accuracy(self, translation_pairs: list[tuple[str, str, int]]):
        correct = 0
        missing = []

        for source_word, target_word, confidence in translation_pairs:
            try:
                if source_word in self.translation_cache:
                    translations = self.translation_cache[source_word]
                else:
                    translations = self.en2nl(source_word)
                    self.translation_cache[source_word] = translations

                if target_word in translations:
                    correct += 1
            except KeyError:
                missing.append((source_word, target_word))

        # if missing:
        #     missing = sorted(missing, key=lambda x: x[0])
        #     print(f"Missing {len(missing)} words from the test set (out of {len(translation_pairs)} words) written to file.")
        #     with open("data/missing.txt", "w", encoding="utf-8") as f:
        #         for source_word, target_word in missing:
        #             sentences = [pair for pair in self.test_data if source_word in [self.preprocess(w) for w in pair[0].split()]]
        #             f.write(f"{source_word} {target_word} {str(sentences)}\n")

        if len(translation_pairs) == len(missing):
            return 0

        return correct / (len(translation_pairs) - len(missing))
    
    def plot_results(self):
        x, y = zip(*self.test_results)
        plt.plot(x, y)
        plt.title(f"Performance over time ({self.name})")
        plt.xlabel("Number of words processed")
        plt.ylabel("Accuracy over all dataset words")
        plt.show()

    
    def show_translations(self):
        translation_pairs = [(source_word, *self.get_most_likely_translation_bayes(source_word)) for source_word in self.data.source_unique_words]
        translation_pairs.sort(key=lambda x: x[2], reverse=False)

        for source_word, target_word, confidence in translation_pairs:
            print(f"[ {source_word} ► {target_word} ] (confidence: {confidence})")
    


# model_p = CrossSituationalModel("Progressive Input Complexity", BilingualDataset(shuffled=False))
# model_p.train()
# model_p.plot_results()

model_r = CrossSituationalModel("Random Input Complexity", BilingualDataset(shuffled=True))
model_r.train()
model_r.plot_results()