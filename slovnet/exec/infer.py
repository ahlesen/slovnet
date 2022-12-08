
from slovnet.record import Record
from slovnet.token import tokenize
from slovnet.markup import (
    BIOMarkup,
    MorphMarkup,
    SyntaxMarkup
)

from .mask import split_masked
import numpy as np

class Infer(Record):
    __attributes__ = ['model', 'encoder', 'decoder']

######
#
#   TAG
#
#####


class TagDecoder(Record):
    __attributes__ = ['tags_vocab']

    def __call__(self, preds):
        for pred in preds:
            yield [self.tags_vocab.decode(_) for _ in pred]


def text_words(text):
    return [_.text for _ in tokenize(text)]

def softmax(z):
    shape = len(z.shape)
    if shape == 2:
        s = z.max(axis=1)
        s = s.reshape(-1, 1)  # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div.reshape(-1, 1)  # dito
        return e_x / div
    elif shape == 1:
        e_x = np.exp(z - np.max(z))
        return e_x / e_x.sum()
    else:
        raise AssertionError("Shape array for softmax bad:", shape)

class NERInfer(Infer):
    def process(self, inputs):
        ### как дб
        for input in inputs:
            # input = input.to(self.model.device)
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            yield from self.model.ner.crf.decode(pred, ~input.pad_mask)

    def process_w_prob(self, inputs):
        preds = []
        probas = []
        for input in inputs:
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            proba = softmax(pred[0])
            pred = self.model.head.crf.decode(pred, ~input.pad_mask)
            preds.append(pred)
            probas.append(proba)
        return preds, probas 


    def __call__(self, texts):
        items = [text_words(_) for _ in texts]
        inputs = self.encoder(items)
        # preds = self.process(inputs)
        preds1, probas = self.process_w_prob(inputs)
        preds1, probas = preds1[0], probas[0]
        probas = [[proba1[pred1] for pred1, proba1 in zip(preds1[0], probas)]]
        preds = self.decoder(preds1)
        for text, item, pred, proba in zip(texts, items, preds, probas):
            tuples = zip(item, pred, proba)
            markup = BIOMarkup.from_tuples(tuples)
            yield markup.to_span(text)


class MorphInfer(Infer):
    def process(self, inputs):
        for input in inputs:
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            pred = self.model.head.decode(pred)
            yield from split_masked(pred, ~input.pad_mask)

    def __call__(self, items):
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            tuples = zip(item, pred)
            yield MorphMarkup.from_tuples(tuples)


########
#
#   SYNTAX
#
######


class SyntaxDecoder(Record):
    __attributes__ = ['rels_vocab']

    def __call__(self, preds):
        for pred in preds:
            head_ids, rel_ids = pred
            ids = [str(_ + 1) for _ in range(len(head_ids))]
            head_ids = [str(_) for _ in head_ids.tolist()]
            rels = [self.rels_vocab.decode(_) for _ in rel_ids]
            yield ids, head_ids, rels


class SyntaxInfer(Infer):
    def process(self, inputs):
        for input in inputs:
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            mask = ~input.pad_mask

            head_id = self.model.head.decode(pred.head_id, mask)
            head_id = split_masked(head_id, mask)

            rel_id = self.model.rel.decode(pred.rel_id, mask)
            rel_id = split_masked(rel_id, mask)

            yield from zip(head_id, rel_id)

    def __call__(self, items):
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            ids, head_ids, rels = pred
            tuples = zip(ids, item, head_ids, rels)
            yield SyntaxMarkup.from_tuples(tuples)
