# -*- coding: utf-8 -*-
# Time    : 2024/5/9
# By      : Yang

import os
from argparse import ArgumentParser
from typing import List
import sentencepiece as spm


def create_spm_model(args):
    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)
    print(args.model_prefix)
    
    spm.SentencePieceTrainer.Train(' '.join([
        f"--input={args.txt_path} ",
        f"--model_prefix={args.model_prefix} ",
        f"--vocab_size={args.vocab_size} ",
        f"--model_type={args.model_type} ",
        f"--normalization_rule_name={args.norm} ",
        f"--control_symbols=<blank> ",  # for CTC loss
        f"--bos_id={args.sos} --eos_id={args.eos} --unk_id={args.unk} ",
        f"--pad_piece=<ig> --bos_piece=<sos> --eos_piece=<eos> --unk_piece={args.unk_str}",
    ]))


class Tokenizer(object):
    def tokenize(self, text):
        raise NotImplementedError
    
    def detokenize(self, token_ids):
        raise NotImplementedError


class CharTokenizer(Tokenizer):
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.char2idx = {
            "<sos>": 0, "<eos>": 1,
            'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8,
            'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15,
            'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21,
            'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27, ' ': 28
        }
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab = len(self.char2idx)
        self.sos_id = self.char2idx['<sos>']
        self.eos_id = self.char2idx['<eos>']
        self.skipped = set()
    
    def tokenize(self, text) -> List[int]:
        if self.do_lower_case:
            text = text.lower()
        remained = [c for c in text if c in self.char2idx]
        skipped = [c for c in text if c not in self.char2idx]
        if len(skipped) > 0:
            for s in skipped:
                if s not in self.skipped:
                    print(f"Skipped character: {s}")
                    self.skipped.add(s)
        return [self.char2idx[char] for char in remained]
    
    def detokenize(self, token_ids) -> str:
        remained = [d for d in token_ids if d in self.idx2char]
        skipped = [d for d in token_ids if d not in self.idx2char]
        if len(skipped) > 0:
            print(f"Skipped token ids: {skipped}")
        return ''.join([self.idx2char[d] for d in remained])


class SubwordTokenizer(Tokenizer):
    def __init__(self, src_path: str, case: str = "upper"):
        """ sub-word tokenizer based on SentencePiece
        Args:
            src_path: the path of SentencePiece model
        """
        assert os.path.exists(src_path)
        assert case in ("upper", "lower", "none")
        self.case = case
        self.src_path = src_path
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(src_path)
        
        self.vocab = self.tokenizer.vocab_size()
        self.sos_id = self.tokenizer.bos_id()
        self.eos_id = self.tokenizer.eos_id()
        self.pad_id = self.tokenizer.pad_id()
        assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()
        
        self.state = f"{self.__class__.__name__}({self.src_path}, case={self.case}, vocab={self.vocab})"
    
    def __repr__(self):
        return self.state
    
    def _convert_case(self, s: str) -> str:
        if self.case == "upper":
            return s.upper()
        elif self.case == "lower":
            return s.lower()
        elif self.case == "none":
            return s
        else:
            raise ValueError(f"Invalid case: {self.case}")
    
    def tokenize(self, text: str) -> List[int]:
        text = self._convert_case(text)
        tokens = self.tokenizer.encode(text, out_type=int)
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        skipped = [t for t in token_ids if t >= self.vocab]
        if len(skipped) > 0:
            print(f"\nTokens not in vocab: {skipped}\n")
        token_ids = [t for t in token_ids if t < self.vocab]
        pieces = [self.tokenizer.id_to_piece(t) for t in token_ids]
        text = ''.join(pieces).replace('▁', ' ').strip()
        text = self._convert_case(text)
        return text


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--txt_path", type=str, help="The path to the training data file")
    parser.add_argument("--model_prefix", type=str, help="Outputs the model name prefix")
    parser.add_argument("--vocab_size", type=int, default=1000, help="The size of the trained vocabulary")
    parser.add_argument("--model_type", type=str, default="bpe", help="model type")
    parser.add_argument("--sos", type=int, default=0, help="start of sentence token id")
    parser.add_argument("--eos", type=int, default=1, help="end of sentence token id")
    parser.add_argument("--unk", type=int, default=3, help="identity by default (no normalization)")
    parser.add_argument("--norm", type=str, default="identity", help="identity by default (no normalization)")
    parser.add_argument("--unk_str", type=str, default=chr(ord('a') + 72),
                        help="identity by default (no normalization)")
    args = parser.parse_args()
    
    # LibriSpeech dataset
    args.txt_path = "../../../data/LibriSpeech/train-clean-100.text"
    args.model_prefix = "./spm/librispeech/{}_{}".format(args.vocab_size, args.model_type)
    
    # LRS2 dataset
    # args.model_prefix ="../../../data/LRS2/train.text"
    # args.txt_path ="./spm/lrs2/{}_{}".format(args.vocab_size,args.model_type)
    is_exist_spm = True
    isDiyTokenizer = True
    
    if isDiyTokenizer:
        # create charTokenizer
        char_tokenizer = CharTokenizer()
        print('char_tokenizer(hello world): char --> idx \n', char_tokenizer.tokenize('hello world'))
        print('char_detokenizer(idx): idx --> char \n',
              char_tokenizer.detokenize([9, 6, 13, 13, 16, 28, 24, 16, 19, 13, 5]))
        
        # create subwordTokenizer base spm
        sub_tokenizer = SubwordTokenizer(args.model_prefix + '.model')
        print('sub_tokenizer(hello world): char --> idx \n', sub_tokenizer.tokenize('hello world'))
        print('sub_detokenizer(idx): idx --> char \n',
              sub_tokenizer.detokenize([35, 43, 977, 674]))
    else:
        if not is_exist_spm:
            create_spm_model(args)
        else:
            spm_tokenizer = spm.SentencePieceProcessor(args.model_prefix + '.model')
            print('spm tokenizer (This is a test, out_type=int): \n', spm_tokenizer.encode('This is a test'))
            print('spm tokenizer (This is a test, out_type=str): \n',
                  spm_tokenizer.encode('This is a test', out_type=str))
