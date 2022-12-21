from __future__ import annotations
from typing import Dict, List, TypeVar, Generic

from genepro.node import Node

T = TypeVar("T")


class Storage(Generic[T]):
    def __init__(self) -> None:
        super().__init__()
        self.__cache: Dict[int, T] = {}

    def get(self, individual: Node) -> T:
        if not self.has_value(individual):
            raise ValueError("The provided individual has not been evaluated yet.")
        return self.__cache[hash(individual)]

    def get_by_hash(self, h: int) -> T:
        if h not in self.__cache:
            raise ValueError("The provided hash is not in the storage dictionary.")
        return self.__cache[h]

    def has_value(self, individual: Node) -> bool:
        h: int = hash(individual)
        return h in self.__cache

    def set_value(self, individual: Node, value: T) -> None:
        h: int = hash(individual)
        self.__cache[h] = value

    def remove(self, individual: Node) -> None:
        if self.has_value(individual):
            del self.__cache[hash(individual)]

    def size(self) -> int:
        return len(self.__cache)

    def keys(self) -> List[int]:
        return list(self.__cache.keys())
