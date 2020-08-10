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
_special = '-'
_punctuation = '！、‘’（），。：；“”？《》'
_arpabet = valid_symbols

_phonemes = '''a1 a2 a3 a4 a5 aa
ai1 ai2 ai3 ai4 ai5
an1 an2 an3 an4 an5
ang1 ang2 ang3 ang4 ang5
ao1 ao2 ao3 ao4 ao5
b c ch d
e1 e2 e3 e4 e5 ee
ei1 ei2 ei3 ei4 ei5
en1 en2 en3 en4 en5
eng1 eng2 eng3 eng4 eng5
er2 er3 er4 er5
f g h
i1 i2 i3 i4 i5
ia1 ia2 ia3 ia4 ia5
ian1 ian2 ian3 ian4 ian5
iang1 iang2 iang3 iang4 iang5
iao1 iao2 iao3 iao4 iao5
ie1 ie2 ie3 ie4 ie5 ii
in1 in2 in3 in4 in5
ing1 ing2 ing3 ing4 ing5
iong1 iong2 iong3 iong4 iong5
iu1 iu2 iu3 iu4 iu5
ix1 ix2 ix3 ix4 ix5
iy1 iy2 iy3 iy4 iy5
iz1 iz2 iz3 iz4 iz5
j k l m n
o1 o2 o3 o4 o5
ong1 ong2 ong3 ong4 ong5
oo ou1 ou2 ou3 ou4 ou5
p q r s sh t
u1 u2 u3 u4 u5
ua1 ua2 ua3 ua4 ua5
uai1 uai2 uai3 uai4 uai5
uan1 uan2 uan3 uan4 uan5
uang1 uang2 uang3 uang4 uang5
ueng1 ueng3 ueng4 ueng5
ui1 ui2 ui3 ui4 ui5
un1 un2 un3 un4 un5
uo1 uo2 uo3 uo4 uo5 uu
v1 v2 v3 v4 v5
van1 van2 van3 van4 van5
ve1 ve2 ve3 ve4 ve5
vn1 vn2 vn3 vn4 vn5 vv
x z zh'''

symbols = [_pad, _eos, _space, _special] + _phonemes.replace('\n', ' ').split(' ') + _arpabet + list(_punctuation)

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


class ChineseProcessor_Phoneme(object):
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
            self.speaker_name = os.path.basename(root_path).split('_')[0]
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
    result = []
    space = _symbol_to_id[_space]
    for s in symbols.split(' '):
        if _should_keep_symbol(s):
            result.append(_symbol_to_id[s])
            result.append(space)
    return result


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


def get_arpabet(cmudict, word):
    arpabet = cmudict.lookup(word)
    return '%s' % arpabet[0] if arpabet is not None else word


def get_phoneme(thchsdict, pinyin):
    phoneme = thchsdict.lookup(pinyin)
    return '%s' % phoneme if phoneme is not None else pinyin
