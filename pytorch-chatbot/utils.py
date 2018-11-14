import torch
import re
import vocab
from cachetools import cached, TTLCache
from cachetools.keys import hashkey

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

ONE_HOUR = 60 * 60


# Lowercase and remove non-letter characters
def normalize_string(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Takes string sentence, returns sentence of word indexes
def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [vocab.EOS_token]


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=10):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


@cached(
    cache=TTLCache(maxsize=int(2 ** 15 - 1), ttl=24*ONE_HOUR),
    key=lambda sent, enc, dec, sear, voc: tuple(hashkey(sent)),
)
def reply_on_sentence(sentence, encoder, decoder, searcher, voc):
    # Normalize sentence
    input_sentence = normalize_string(sentence)
    # Evaluate sentence
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == "EOS" or x == "PAD")]
    return " ".join(output_words)
