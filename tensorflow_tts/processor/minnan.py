# -*- coding: utf-8 -*-
# Copyright 2020 YSJ@TalentedSoft

import os
import soundfile as sf
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Union, Tuple, Any
from tensorflow_tts.processor import BaseProcessor
from tensorflow_tts.processor.minnan_frontend import text2phoneme, minnan_phoneme


_pad = ["pad"]
_eos = ["eos"]
_pause = ["sp", "np", "lp"]

MINNAN_SYMBOLS = _pad + _pause + minnan_phoneme + _eos

pause_punctuation = {
    "？": "np",
    "！": "np",
    "，": "np",
    "。": "np",
    "：": "np",
    "、": "np",
    "；": "np",
    ",": "np",
    ".": "np",
    "?": "np",
    "!": "np",
}

@dataclass
class MinNanProcessor(BaseProcessor):
    """MinNan dataset process"""
    symbols: List[str] = field(default_factory=lambda: MINNAN_SYMBOLS)
    cleaner_names: str = None

    def __post_init__(self):
        super().__post_init__()
        self.text2phoneme = text2phoneme
        self.pause_punctuation = pause_punctuation

    def create_items(self):
        if self.data_dir:
            with open(os.path.join(self.data_dir, self.train_f_name), encoding='utf-8') as f:
                self.items = [self.split_line(self.data_dir, line, self.delimiter) for line in f]
    
    def split_line(self, data_dir, line, split):
        content = line.strip().split(split)
        wave_file = content[self.positions["file"]]
        text = content[self.positions["text"]]
        speaker_name = content[self.positions["speaker_name"]]
        wav_path = os.path.join(self.data_dir, 'wavs', f"{wave_file + self.f_extension}")
        return text, wav_path, speaker_name

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item
        
        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_path)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text, mode='training'), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_path)[-1].split(".")[0],
            "speaker_id": self.get_speaker_id(speaker_name),
            "rate": rate
        }
        return sample

    def setup_eos_token(self):
        return _eos[0]

    def text_to_sequence(self, text, mode='inference', show_phoneme=False, input_is_pinyin=True):
        if mode == 'inference':
            phoneme = self.text2phoneme(text, show_wordsegment=show_phoneme)
            if show_phoneme:
                print(f"Convert to : \n {phoneme}")
            phoneme = self.punctuation2silence(phoneme)
            phoneme = 'sp ' + phoneme
            if show_phoneme:
                print(f"Convert punctuation to silence : \n {phoneme}")
        elif mode == 'phoneme':
                return self.remove_punctuation(text)
        else:
            phoneme = text

        sequence = []
        sequence += self.symbols_to_sequence(phoneme)
        # add eos tokens
        sequence += [self.eos_id]

        return sequence
    
    def symbols_to_sequence(self, symbols):
        result = []
        for s in symbols.split(' '):
            if self.should_keep_symbol(s):
                result.append(self.symbol_to_id[s])
        return result

    def should_keep_symbol(self, s):
        return s in self.symbol_to_id and s is not "pad" and s is not "eos"

    def remove_punctuation(self, phoneme):
        result = []
        for p in phoneme.split(' '):
            if self.should_keep_symbol(p):
                result.append(p)
        return ' '.join(result)
        
    def punctuation2silence(self, phoneme):
        result = []
        for p in phoneme.split(' '):
            sil = self.pause_punctuation.get(p) 
            p = sil if sil is not None else p
            if self.should_keep_symbol(p):
                result.append(p)
        return ' '.join(result)
