# -*- coding: utf-8 -*-
# Copyright 2020 YSJ@TalentedSoft

import pypinyin as py
import re
from .cn_tn import NSWNormalizer
split_character = ['，', '。', '？', '！', ' ', '、', '；', '：']
_whitespace_re = re.compile(r'\s+')


def text2pinyin(text):
    text = collapse_whitespace(text)
    text = NSWNormalizer(text).normalize()
    text = special_symbol_clean(text)
    text = text.upper()
    pinyin = get_pinyin(text)
    return pinyin


def get_pinyin(text):
    pinyin = ""
    pinyin_list = py.pinyin(text, style=py.Style.TONE3)
    for content in pinyin_list:
        str = ''.join(content)
        pinyin += str + ' '
    return pinyin[:-1]


def text_segment(text, position=200, max_len=200, split_text=[]):
    if position >= len(text):
        if text[position - max_len:] != '':
            split_text.append(text[position - max_len:])
        return split_text
    else:
        for i in range(position, position - max_len, -1):
            if i == position - max_len + 1:
                split_text.append(text[position - max_len:position])
                return text_segment(text, position + max_len, max_len, split_text)
            if text[i] in split_character:
                split_position = i + 1
                split_text.append(text[position - max_len:split_position])
                return text_segment(text, split_position + max_len, max_len, split_text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def special_symbol_clean(text):
    text = text.replace('www.', '三W点')
    text = text.replace('.', '点')
    text = text.replace('@', 'at ')
    # text = text.replace('-', '杠')
    return text
