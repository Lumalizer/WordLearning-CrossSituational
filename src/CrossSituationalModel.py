from collections import defaultdict
import time
import numpy as np
from BilingualDataset import BilingualDataset
from word2word import Word2word
import matplotlib.pyplot as plt
import tqdm
import os

class CrossSituationalModel:
    def __init__(self, name, data: BilingualDataset, interval_method="uniquewords"):
        os.makedirs("results", exist_ok=True)

        self.name = name
        self.interval_method = interval_method
        self.en2nl = Word2word("en", "nl")

        self.data = data
        self.source_word_counts = defaultdict(int)
        self.target_word_counts = defaultdict(int)

        self.source_unique_words = set()
        self.target_unique_words = set()
        
        self.table = defaultdict(lambda: defaultdict(int))
        self.translations = defaultdict(str)
        self.results = None
    
    def process_sentence_pair(self, pair: tuple[str, str]):
        source_sentence, target_sentence = pair
        source_words = source_sentence.split()
        target_words = target_sentence.split()

        for source_word in source_words:
            for target_word in target_words:
                self.table[source_word][target_word] += 1
                self.source_word_counts[source_word] += 1
                self.target_word_counts[target_word] += 1

        self.source_unique_words.update(source_words)
        self.target_unique_words.update(target_words)

    def get_most_likely_translation(self, source_word: str):
        target_words = np.array(list(self.table[source_word].keys()))
        smooth = 1

        if len(target_words) == 0:
            return "", 0

        table_counts = np.array([self.table[source_word][word] for word in target_words])
        target_word_counts = np.array([self.target_word_counts[word] for word in target_words])

        probabilities = (table_counts + smooth) / (target_word_counts + smooth * len(target_words))

        max_index = np.argmax(probabilities)
        return target_words[max_index], probabilities[max_index]

    def train(self):
        results = [(0, 0, 0)]
        processed = 0
        total_processed = 0
        interval_method = self.interval_method

        if interval_method == "totalwords":
            len_interval = self.data.source_words_total // 100
        elif interval_method == "uniquewords":
            len_interval = len(self.data.source_unique_words) // 100


        for i, pair in tqdm.tqdm(enumerate(self.data), total=len(self.data)):
            if interval_method == "totalwords":
                processed += len(pair[0].split())
                total_processed += len(pair[0].split())
            elif interval_method == "uniquewords":
                processed += len ([word for word in set(pair[0].split()) if word not in self.source_unique_words])

            self.process_sentence_pair(pair)

            if processed < len_interval and i != len(self.data) - 1:
                continue

            evaluation_results = self.test()
            results.append((total_processed if interval_method == "totalwords" else len(self.source_unique_words), *evaluation_results))
            tqdm.tqdm.write(f"Test performance (pairs processed={i}): {evaluation_results}") 
            processed -= len_interval

        self.results = results

        with open(f"results/{time.time()}{self.name}.txt", 'w') as f:
            f.write(str(results))

        print(len(results))
        return results

    def test(self):
        encountered = set(self.source_word_counts.keys())

        translation_pairs = [(source_word, *self.get_most_likely_translation(source_word)) for source_word in encountered]
        words_learned, accuracy = self.evaluate(translation_pairs)

        return words_learned, accuracy

    def evaluate(self, translation_pairs: list[tuple[str, str, int]]):
        correct = 0
        wrong = 0
        missing = []

        for source_word, target_word, confidence in translation_pairs:
            try:
                translations = self.en2nl(source_word)
                if target_word in translations:
                    correct += 1
                else:
                    wrong += 1
            except KeyError:
                missing.append((source_word, target_word))

        if correct + wrong == 0:
            return 0

        return correct, correct / (correct + wrong)
    
    def plot_results(self, other_model:'CrossSituationalModel' = None, difference=False):
        x, words_learned, accuracy = zip(*self.results)

        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
        # fig.tight_layout(pad=2.3)
        fig.subplots_adjust(left=0.2)
        ax1: plt.Axes
        ax2: plt.Axes
        ax2.set_xlabel(f'{"Unique words" if self.interval_method == "uniquewords" else "Total words"} processed')

        if other_model is None or not difference:
            fig.suptitle(f"Results for {self.name}")
            ax1.set_ylabel("Words learned")
            ax1.set_ylim([0, len(self.data.source_unique_words)])
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim([0, 1])

        if other_model is not None:
            assert self.interval_method == other_model.interval_method

            x_, words_learned_, accuracy_ = zip(*other_model.results)

            if difference:
                words_learned = np.array(words_learned) / np.array(words_learned_)
                accuracy = np.array(accuracy) - np.array(accuracy_)
                fig.suptitle(f"{self.name} vs {other_model.name} Input Complexity (difference)")
                ax1.set_ylabel(f"Unique words learned\n({self.name} / {other_model.name})")
                ax2.set_ylabel(f"Accuracy\n({self.name} - {other_model.name})")
            else:
                fig.suptitle(f"{self.name} vs {other_model.name} Input Complexity")
                ax1.set_ylabel("Unique words learned")
                ax2.set_ylabel("Accuracy")
                ax1.plot(x_, words_learned_, label=other_model.name)
                ax2.plot(x_, accuracy_, label=other_model.name)

        ax1.plot(x, words_learned, label=self.name if not difference else None)
        ax2.plot(x, accuracy, label=self.name if not difference else None)

        if other_model is not None and not difference:
            ax1.legend()
            ax2.legend()

        plt.savefig(f"results/{time.time()}{self.name}.png")
        plt.show()
    
    def show_translations(self):
        translation_pairs = [(source_word, *self.get_most_likely_translation(source_word)) for source_word in self.source_unique_words]
        translation_pairs.sort(key=lambda x: x[2], reverse=False)

        for source_word, target_word, confidence in translation_pairs:
            print(f"[ {source_word} ► {target_word} ] (confidence: {confidence})")
