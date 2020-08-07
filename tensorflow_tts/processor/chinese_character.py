# -*- coding: utf-8 -*-
# modified by ysj（@unparalleled-ysj）

import re
import os

import numpy as np
import soundfile as sf

from tensorflow_tts.utils import cleaners
from tensorflow_tts.processor.cmudict import CMUDict, valid_symbols
from tensorflow_tts.processor.thchsdict import THCHSDict

_pad = "_"
_eos = "~"
_space = ' '
_punctuation = '！、‘’（），。：；“”？《》'
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in valid_symbols]

# Export all symbols:
symbols = [_pad, _eos, _space] + list(_special) + list(_punctuation) + list(_letters)  # + _arpabet

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


class ChineseProcessor_Character(object):
    '''Chinese processor'''

    def __init__(self, root_path, cleaner_names):
        self.root_path = root_path
        self.cleaner_names = cleaner_names

        # initial chinese and engish phoneme dict
        thchsdict_path = os.path.abspath(r'tensorflow_tts/dictionary/thchs_tonebeep')
        self.thchsdict = THCHSDict(thchsdict_path)
        cmudict_path = os.path.abspath(r'tensorflow_tts/dictionary/cmudict-0.7b')
        self.cmudict = CMUDict(cmudict_path)

        items = []
        if root_path is not None:
            self.speaker_name = os.path.basename(root_path)
            for root, _, files in os.walk(os.path.join(root_path, 'data')):
                for f in files:
                    if f.endswith('.trn'):
                        trn_file = os.path.join(root, f)
                        with open(trn_file, encoding='utf-8')as f:
                            wav_path = trn_file[:-4]
                            text = f.readline().strip()
                            items.append([text, wav_path, self.speaker_name])

            self.items = items


    def get_one_sample(self, item):
        text, wav_file, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_file)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_file)[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text, show_phoneme=False):

        sequence = []
        pinyin = text
        text = [get_phoneme(self.thchsdict, word) for word in text.split(' ')]
        text = ' '.join([get_arpabet(self.cmudict, word) for word in text])
        if show_phoneme:
            print(f"{pinyin} convert to : {text}")
        sequence += _symbols_to_sequence(_clean_text(text, [self.cleaner_names]))

        return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


def get_arpabet(cmudict, word):
    arpabet = cmudict.lookup(word)
    return '%s' % arpabet[0] if arpabet is not None else word


def get_phoneme(thchsdict, pinyin):
    phoneme = thchsdict.lookup(pinyin)
    return '%s' % phoneme if phoneme is not None else pinyin
