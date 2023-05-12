from __future__ import annotations
from typing import TypeVar, Generic
from weakref import WeakKeyDictionary
from genepro.node import Node

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class Storage(Generic[T]):
    def __init__(self) -> None:
        super().__init__()
        self.__cache: dict[int, T] = {}

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

    def keys(self) -> list[int]:
        return list(self.__cache.keys())


class WeakCache:
    def __init__(self) -> None:
        super().__init__()
        self.__cache: WeakKeyDictionary = WeakKeyDictionary()
        self.__hit: int = 0
        self.__tot: int = 0

    def contains(self, key: K) -> bool:
        return key in self.__cache

    def get_hit(self) -> int:
        return self.__hit

    def get_tot(self) -> int:
        return self.__tot

    def get_miss(self) -> int:
        return self.get_tot() - self.get_hit()
    
    def hit_ratio(self) -> float:
        return self.get_hit() / float(self.get_tot())
    
    def miss_ratio(self) -> float:
        return self.get_miss() / float(self.get_tot())
    
    def empty_cache(self) -> None:
        self.__cache = WeakKeyDictionary()
        self.__tot = 0
        self.__hit = 0

    def cache_size(self) -> int:
        return len(self.__cache)
    
    def cache_keys(self) -> list[K]:
        return sorted(list(self.__cache.keys()), key=lambda x: hash(x))
    
    def get(self, key: K) -> V:
        self.__tot += 1
        if self.contains(key):
            self.__hit += 1
            return self.__cache[key]
        return None

    def set(self, key: K, value: V) -> None:
        self.__cache[key] = value
    