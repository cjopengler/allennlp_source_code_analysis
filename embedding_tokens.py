#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-

#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2018/12/18 09:39:00
"""

from allennlp.data.fields import TextField
from allennlp.data import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data import Token

words = ["All", "the", "cool", "kids", "use", "character", "embeddings", "."]
sentence1 = TextField(
    tokens=[Token(w) for w in words],
    token_indexers={
        'tokens': SingleIdTokenIndexer(namespace='token_ids'),
        'characters': TokenCharactersIndexer(namespace='token_characters')
    }
)

words2 = ["I", "prefer", "word2vec", "though", "..."]
sentence2 = TextField(
    tokens=[Token(w) for w in words2],
    token_indexers={
        'tokens': SingleIdTokenIndexer(namespace='token_ids'),
        'characters': TokenCharactersIndexer(namespace='token_characters')
    }
)

instance1 = Instance({'sentence': sentence1})
instance2 = Instance({'sentence': sentence2})

from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch

instances = [instance1, instance2]

vocab = Vocabulary.from_instances(instances)
print('This is the token_ids vocabuary we created')
print(vocab.get_index_to_token_vocabulary('token_ids'))

print('Characters we create')
print(vocab.get_index_to_token_vocabulary('token_characters'))

# 输出index instance

for instance in instances:
    instance.index_fields(vocab)

from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

# word embedding
word_embedding = Embedding(
    num_embeddings=vocab.get_vocab_size('token_ids'),
    embedding_dim=10
)

char_embedding = Embedding(
    num_embeddings=vocab.get_vocab_size('token_characters'),
    embedding_dim=5
)

character_cnn = CnnEncoder(embedding_dim=5, num_filters=2, output_dim=8)

token_character_encoder = TokenCharactersEncoder(embedding=char_embedding,
                                                 encoder=character_cnn)
text_filed_embedder = BasicTextFieldEmbedder({
    'tokens': word_embedding,
    'characters': token_character_encoder
})

from allennlp.data.dataset import Batch

batch = Batch(instances)

tensors = batch.as_tensor_dict(batch.get_padding_lengths())

print('Torch tensors for passing to a model\n', tensors)
print('\n\n')

print('Begin: text_filed_viaribles: ----------------------')
text_filed_viaribles = tensors['sentence']
print(text_filed_viaribles)
print('End: text_filed_viaribles: ----------------------')

embedded_text = text_filed_embedder(text_filed_viaribles)

dimensions = list(embedded_text.size())

print('Post embedding with our TextFiled Embedder')
print('Batch size: ', dimensions[0])
print('Sentence Length: ', dimensions[1])
print('Embeeding size: ', dimensions[2])



