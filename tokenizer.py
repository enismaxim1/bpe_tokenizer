from pathlib import Path
import pickle
import string
from typing import List, Optional, Set
from collections import defaultdict
import os
import re

class Tokenizer:

    def __init__(self, path: Optional[os.PathLike] = None):
        self.token_to_ids = dict()
        self.id_to_tokens = dict()
        if path:
            self.train(Path(path))

    def _next_token(self, text: str, start: int):
        # TODO: improve performance with a Trie
        if text[start] not in self.token_to_ids:
            return None
        curr_end = start + 1
        latest_token = text[start:curr_end]
        while curr_end <= len(text) and any(text[start:curr_end] in token for token in self.token_to_ids):
            if text[start:curr_end] in self.token_to_ids:
                latest_token = text[start:curr_end]
            curr_end += 1
        return latest_token

    def is_known(self, char: str):
        # TODO: add unknown characters (?)
        return True

    def train(self, data: Path, max_token_limit = 500):
        # TODO: extremely slow O(n^2) - can performance be improved on large texts?
        # TODO: smarter handling of punctuation, etc
        # TODO: correctly process data which contains special tokens
        if self.is_cached(data):
            try:
                token_to_ids, id_to_tokens = self.retrieve_cache(data)
                self.token_to_ids = token_to_ids
                self.id_to_tokens = id_to_tokens
                return
            except ValueError:
                # Backwards compatibility to handle cached iterable tokens
                tokens = self.retrieve_cache(data)
                token_to_ids, id_to_tokens = {}, {}
                for i, token in enumerate(tokens):
                    token_to_ids[i] = token
                    id_to_tokens[token] = i
                self.token_to_ids = token_to_ids
                self.id_to_tokens = id_to_tokens
                return

        text = data.read_text()
        UNKNOWN = "[UNK]"

        def skip_join(token: str):
            # TODO: handle punctation and stuff here
            return token == UNKNOWN or token.isspace()

        token_to_ids = dict()
        tokenized_text = []
        for char in text:
            if self.is_known(char):
                token = char
            else:
                token = UNKNOWN
            if token not in token_to_ids:
                token_to_ids[token] = len(token_to_ids)
            tokenized_text.append(token)
        
        while(len(token_to_ids) < max_token_limit):
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
            token_to_ids[new_token] = len(token_to_ids)
            print(new_token, len(token_to_ids))
        
        self.token_to_ids = token_to_ids
        self.id_to_tokens = {token_id: token for token, token_id in token_to_ids.items()}
        self.cache_tokens(data, self.token_to_ids, self.id_to_tokens)

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

    def cache_tokens(self, text_path: Path, token_to_ids, id_to_tokens):
        cache_path = self.get_cache_path(text_path)
        if not cache_path.exists():
            cache_path.touch()
            with open(str(cache_path), 'wb') as f:
                pickle.dump((token_to_ids, id_to_tokens), f)

    def is_cached(self, text_path: Path):
        return self.get_cache_path(text_path).exists()

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

    def tokenize_to_ids(self, text: str):
        text_tokenization = self.tokenize(text)
        return [self.token_to_ids[token] for token in text_tokenization]

    def ids_to_text(self, token_ids: List[str]):
        return "".join(self.id_to_tokens[token_id] for token_id in token_ids)
    

t = Tokenizer('res/small_book.txt')
print(t.tokenize("a small green cat ate nothing"))
ids = t.tokenize_to_ids("a small green cat ate nothing, but the itsy bitsy spider ate something")
print(ids)
print(t.ids_to_text(ids))