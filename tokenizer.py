from pathlib import Path
import pickle
import string
from typing import List, Optional, Set
from collections import defaultdict
import os
import re

class Tokenizer:

    def __init__(self, path: Optional[os.PathLike] = None):
        self.tokens = set()
        if path:
            self.train(Path(path))

    def _next_token(self, text: str, start: int):
        # TODO: improve performance with a Trie
        if text[start] not in self.tokens:
            return None
        curr_end = start + 1
        latest_token = text[start:curr_end]
        while curr_end <= len(text) and any(text[start:curr_end] in token for token in self.tokens):
            if text[start:curr_end] in self.tokens:
                latest_token = text[start:curr_end]
            curr_end += 1
        return latest_token

    def is_known(self, char: str):
        # TODO: add unknown characters (?)
        return True

    def train(self, data: Path, max_token_limit = 1000):
        # TODO: extremely slow O(n^2) - can performance be improved on large texts?
        # TODO: smarter handling of punctuation, etc

        if (tokens := self.retrieve_cache(data)):
            self.tokens = tokens
            return

        text = data.read_text()
        UNKNOWN = "[UNK]"

        def skip_join(token: str):
            # TODO: handle punctation and stuff here
            return token == UNKNOWN or token.isspace()

        tokens = set()
        tokenized_text = []
        for char in text:
            if self.is_known(char):
                token = char
            else:
                token = UNKNOWN
            tokens.add(token)
            tokenized_text.append(token)
        
        while(len(tokens) < max_token_limit):
            token_freqs = defaultdict(int)
            for i in range(len(tokenized_text) - 1):
                if skip_join(tokenized_text[i]) or skip_join(tokenized_text[i+1]):
                    continue
                token_pair = (tokenized_text[i], tokenized_text[i+1])
                token_freqs[token_pair] += 1

            (token1, token2) = max(token_freqs, key= token_freqs.get)
            # break if no further compression is possible
            if token_freqs[(token1, token2)] <= 1:
                break

            new_token = token1 + token2
            i = 0
            while i < len(tokenized_text) - 1:
                token_pair = tokenized_text[i] + tokenized_text[i+1]
                if token_pair == new_token:
                    tokenized_text[i] = token_pair
                    del tokenized_text[i+1]
                i += 1
            tokens.add(new_token)
            print(new_token, len(tokens))
        
        self.tokens = tokens
        self.cache_tokens(data, tokens)

    def get_cache_path(self, filename: Path) -> Path:
        # TODO: cache to a better directory
        name = str(filename.resolve())
        # Replace invalid characters with underscores
        valid_filename = re.sub(r'[\\/:*?"<>|]', '_', name)
        # Replace any sequences of multiple underscores with a single underscore
        valid_filename = re.sub(r'__+', '_', valid_filename)
        cache_dir = Path("tokenizer_caches")
        cache_name =  f"{valid_filename}.pkl"
        return cache_dir / cache_name

    def cache_tokens(self, text_path: Path, tokens: List[str]):
        cache_path = self.get_cache_path(text_path)
        if not cache_path.exists():
            cache_path.touch()
            with open(str(cache_path), 'wb') as f:
                pickle.dump(tokens, f)


    def retrieve_cache(self, text_path: Path):
        cache_path = self.get_cache_path(text_path)
        if cache_path.exists():
            with open(str(cache_path), 'rb') as f:
                return pickle.load(f)

    def tokenize(self, text: str):
        tokens = []
        curr_index = 0
        while curr_index < len(text):
            next_token = self._next_token(text, curr_index)
            if next_token == None:
                tokens.append("[UNK]")
                curr_index += 1
                continue
            tokens.append(next_token)
            curr_index += len(next_token)
        return tokens
