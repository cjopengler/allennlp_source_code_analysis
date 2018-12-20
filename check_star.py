#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-

#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2018/12/19 07:53:00
"""

a = [[[1, 2], [3, 4]]]

class A(object):

    def __call__(self, *input, **kwargs):
        print(input)
        print(*input)
        # print(type(*input))
        print(type(input))

A()(*a)