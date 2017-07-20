import numpy as np
import pickle
lines = open("/Users/vidurj/nmt/nmt/testdata/ppdb-1.0-s-phrasal", "r").read().strip().split("\n")
phrase1s = []
phrase2s = []
vocab = set()
pairs = set()
for line in lines:
    tokens = line.split("|||")
    if tokens[0].strip() == "[S]":
        phrase1 = tokens[1].strip()
        phrase2 = tokens[2].strip()
        pair = (phrase1, phrase2)
        if pair in pairs:
            continue
        else:
            pairs.add(pair)
        phrase1s.append(phrase1)
        phrase2s.append(phrase2)
        print(phrase1, phrase2)
        words = set(phrase1.split() + phrase2.split())
        vocab = vocab.union(words)

testPhrase1s = phrase1s[: int(len(phrase1s)/4)]
testPhrase2s = phrase2s[:int(len(phrase2s)/4)]
trainPhrase1s = phrase1s[int(len(phrase1s)/4):]
trainPhrase2s = phrase2s[int(len(phrase2s)/4):]
print(len(phrase1s))
vocab = ["<unk>", "<s>", "</s>"] + list(vocab)
print(len(vocab))
print(vocab)
word_to_index = dict(zip(vocab, range(len(vocab))))
vocab_str = "\n".join(vocab)
open("vocab.a", "w").write(vocab_str)
open("vocab.b", "w").write(vocab_str)
open("train.a", "w").write("\n".join(trainPhrase1s))
open("train.b", "w").write("\n".join(trainPhrase2s))
open("test.a", "w").write("\n".join(testPhrase1s))
open("test.b", "w").write("\n".join(testPhrase2s))


word_vec_data = open("/Users/vidurj/Downloads/glove/glove.6B.100d.txt", "rb").read().strip().split("\n")
vecs = np.random.randn(len(word_to_index), 100)
num_found = 0
for index, point in enumerate(word_vec_data):
    temp = point.split()
    word = temp[0].strip()
    if word in word_to_index:
        num_found += 1
        vecs[word_to_index[word]] = [float(x) for x in temp[1:]]
print("found", num_found)
print(np.shape(vecs))
np.save("vectors", vecs)