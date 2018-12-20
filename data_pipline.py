#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-

#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2018/12/17 09:35:00
"""

from allennlp.data import Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer

review = TextField(tokens=list(map(Token, ['This', 'movie', 'was', 'awful', '!'])),
                   token_indexers={'tokens': SingleIdTokenIndexer(namespace='token_ids')})
review_sentiment = LabelField(label='negative', label_namespace='tags')

print('Tokens in TextFile: ', review.tokens)
print('label of LabelFiled', review_sentiment.label)

from allennlp.data import Instance

instance1 = Instance({'review': review, 'label': review_sentiment})

review2 = TextField(list(map(Token, ["This", "movie", "was", "quite", "slow", "but", "good", ".", "slow", "but"])), token_indexers={"tokens": SingleIdTokenIndexer(namespace="token_ids")})
review_sentiment2 = LabelField("positive", label_namespace="tags")
instance2 = Instance({"review": review2, "label": review_sentiment2})

instances = [instance1, instance2]


from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch

vocab = Vocabulary.from_instances(instances)

print('This is the id->word mappding for the toke_ids namespace')
print(vocab.get_index_to_token_vocabulary('token_ids'))
print('This is the id->word mappding for the tags namespace')
print(vocab.get_index_to_token_vocabulary('tags'))
print('Vocab token to index dictionary: ', vocab._token_to_index)

batch = Batch(instances)

# 将所有的token全部变成index
batch.index_instances(vocab)

padding_lengths = batch.get_padding_lengths()
print('Lnghts used for padding', padding_lengths)

tensor_dict = batch.as_tensor_dict(padding_lengths)
print('tensor_dict', tensor_dict)

