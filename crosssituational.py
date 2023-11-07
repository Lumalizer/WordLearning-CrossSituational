from collections import defaultdict
from dataclasses import dataclass

@dataclass
class BilingualDataset:
    filename: str

    def __post_init__(self):
        with open(self.filename, 'r', encoding="utf-8") as f:
            self.lines = [line.split('\t')[:2] for line in f.readlines()]

    def __iter__(self):
        for line in self.lines:
            yield line[0], line[1]


class CrossSituationalModel:
    def __init__(self, dataset: BilingualDataset = []):
        self.source_word_counts = defaultdict(int)
        self.target_word_counts = defaultdict(int)

        self.source_unique_words = set()
        self.target_unique_words = set()
        
        self.table = defaultdict(lambda: defaultdict(int))

        for pair in dataset:
            self.process_sentence_pair(pair)


    def preprocess(self, text: str):
        for char in [",", ".", "?", "!"]:
            text = text.replace(char, "")
        return text.lower()
    
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
    
    def get_most_likely_translation(self, source_word: str):
        target_words = list(self.table[source_word].keys())
        target_words_score = [self.table[source_word][word] / self.target_word_counts[word] for word in target_words]
        target_words_confidence = [self.table[source_word][word] for word in target_words]
        max_index = target_words_score.index(max(target_words_score))
        return target_words[max_index], target_words_confidence[max_index] - 1
    
    def show_results(self):
        translation_pairs = [(source_word, *self.get_most_likely_translation(source_word)) for source_word in self.source_unique_words]
        translation_pairs.sort(key=lambda x: x[2], reverse=False)

        for source_word, target_word, confidence in translation_pairs:
            print(f"[ {source_word} ► {target_word} ] (confidence: {confidence})")
    
model = CrossSituationalModel(BilingualDataset("data/nld.txt"))
model.show_results()