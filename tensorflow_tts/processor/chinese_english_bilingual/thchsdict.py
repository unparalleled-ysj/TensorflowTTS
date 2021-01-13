# -*- coding: utf-8 -*-
# Copyright 2020 YSJ@TalentedSoft

thchs_phoneme = [
    'a1', 'a2', 'a3', 'a4', 'a5', 'aa', 
    'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 
    'an1', 'an2', 'an3', 'an4', 'an5', 
    'ang1', 'ang2', 'ang3', 'ang4', 'ang5', 
    'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 
    'b', 'c', 'ch', 'd', 
    'e1', 'e2', 'e3', 'e4', 'e5', 'ee', 
    'ei1', 'ei2', 'ei3', 'ei4', 'ei5', 
    'en1', 'en2', 'en3', 'en4', 'en5', 
    'eng1', 'eng2', 'eng3', 'eng4', 'eng5', 
    'er1', 'er2', 'er3', 'er4', 'er5', 
    'f', 'g', 'h', 
    'i1', 'i2', 'i3', 'i4', 'i5', 'ii',
    'ia1', 'ia2', 'ia3', 'ia4', 'ia5', 
    'ian1', 'ian2', 'ian3', 'ian4', 'ian5', 
    'iang1', 'iang2', 'iang3', 'iang4', 'iang5', 
    'iao1', 'iao2', 'iao3', 'iao4', 'iao5', 
    'ie1', 'ie2', 'ie3', 'ie4', 'ie5', 
    'in1', 'in2', 'in3', 'in4', 'in5', 
    'ing1', 'ing2', 'ing3', 'ing4', 'ing5', 
    'iong1', 'iong2', 'iong3', 'iong4', 'iong5', 
    'iu1', 'iu2', 'iu3', 'iu4', 'iu5', 
    'ix1', 'ix2', 'ix3', 'ix4', 'ix5', 
    'iy1', 'iy2', 'iy3', 'iy4', 'iy5', 
    'iz1', 'iz2', 'iz3', 'iz4', 'iz5', 
    'j', 'k', 'l', 'm', 'n', 
    'o1', 'o2', 'o3', 'o4', 'o5', 'oo', 
    'ong1', 'ong2', 'ong3', 'ong4', 'ong5', 
    'ou1', 'ou2', 'ou3', 'ou4', 'ou5', 
    'p', 'q', 'r', 's', 'sh', 't', 
    'u1', 'u2', 'u3', 'u4', 'u5', 'uu', 
    'ua1', 'ua2', 'ua3', 'ua4', 'ua5', 
    'uai1', 'uai2', 'uai3', 'uai4', 'uai5', 
    'uan1', 'uan2', 'uan3', 'uan4', 'uan5', 
    'uang1', 'uang2', 'uang3', 'uang4', 'uang5', 
    'ueng1', 'ueng3', 'ueng4', 'ueng5', 
    'ui1', 'ui2', 'ui3', 'ui4', 'ui5', 
    'un1', 'un2', 'un3', 'un4', 'un5', 
    'uo1', 'uo2', 'uo3', 'uo4', 'uo5', 
    'v1', 'v2', 'v3', 'v4', 'v5', 
    'van1', 'van2', 'van3', 'van4', 'van5', 
    've1', 've2', 've3', 've4', 've5', 
    'vn1', 'vn2', 'vn3', 'vn4', 'vn5', 
    'vv', 'x', 'z', 'zh']


class THCHSDict:
    '''Thin wrapper around THCHSDict data to convert pinyin to phoneme'''
    def __init__(self, file):
        with open(file, encoding='utf-8') as f:
            entries = _parse_thchsdict(f)
        self._entries = entries


    def __len__(self):
        return len(self._entries)


    def lookup(self, pinyin):
        '''Returns list of  phoneme of the given pinyin'''
        return self._entries.get(pinyin)


def _parse_thchsdict(file):
    thchsdict = {}
    for line in file:
        parts = line.strip('\n').split(' ')
        pinyin = parts[0]
        phoneme = ' '.join(parts[1:])
        thchsdict[pinyin] = phoneme
    return thchsdict
