#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-

#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2018/12/17 17:02:00
"""

from allennlp.data import DatasetReader

class WhiteList(DatasetReader):

    def _read(self, file_path: str):

        self.max_len = 0
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                index = line.find('/')
                # print('index: {}'.format(index))
                if index > 0:

                    if self.max_len < index:
                        self.max_len = index

                    if index > 48:
                        print(line[0:index])

                    yield line[0:index]


if __name__ == '__main__':
    wl = WhiteList(lazy=True)

    input_fp = '../data/white_list.bak'
    with open('../data/white_list', 'w') as f:

        for target in wl.read(input_fp):

            f.write(target + '\n')
    print(wl.max_len)


