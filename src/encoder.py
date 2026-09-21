from typing import Any

import torch
import torch.nn as nn

class Encoder:
    def __init__(self, txt: str="../data/input.txt") -> None:
        self.hashset: dict = {"mask": 0}
        # txt: str = "Hello, this is Leon speaking."

        with open(txt) as f:
            lines: list[str] = f.readlines()

        line: str = "\n".join(lines)

        # self.txt = txt
        i: int = 0
        for c in line:
            if c not in self.hashset:
                i += 1
                self.hashset[c] = i
            else:
                continue

    def encode(self, text: str) -> list:
        return [self.hashset[c] for c in text]

    def decode(self, embeddings: list[int]) -> list[str]:
        txt: list[str] = []
        lookup: dict = {v: k for k, v in zip(self.hashset.keys(), self.hashset.values())}
        for e in embeddings:
            txt.append(lookup[e])
        return txt

    def lookuptable(self) -> Any:
        return self.hashset.values()

if __name__ == "__main__":
    encoder: Encoder = Encoder()
    print(encoder.encode("hello"))
    print(encoder.decode(encoder.encode("b")))
    print(encoder.lookuptable())
    print(encoder.hashset)
    print("success")