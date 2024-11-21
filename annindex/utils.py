from bisect import bisect_left
from collections.abc import Callable, Generator, Sequence, Iterator
from typing import TypeVar, Any
from dataclasses import dataclass
from itertools import chain

P = TypeVar('P')
V = TypeVar('V')

def peek_iterator(x: Iterator[P]) -> tuple[P, Iterator[P]]:
    """
    Peeks to top of iterator and return new iterator equivalent to the old one (before peeking).

    Do not use in a loop, it is will result in quadratic complexity.

    Parameters
    ----------
    x : Iterator[P]
        Iterator to peek into

    Returns
    -------
    first : P
        First item in x
    newit : Iterator[P]
        New iterator equivalent to the original x (will yield the first item again)
    """        
    if isinstance(x, Sequence):        
        return x[0], x # if its a list or array, just return first item
    else:
        # Peek one element to check that dimension matches
        gen = iter(x)
        first = next(gen)
        return first, chain([first], gen)

@dataclass(order=True)
class PriorityWrapper():
    """Keep Track of items in queue."""    
    prio: P
    value: V
    visited: bool = False

class VisitPriorityQueue():
    """
    A variant of priority queue with two crucial differences:
    it supports limited size, filters returns to previously unvisited elements, and can prevent double insertions.

    Parameters
    ----------
    maxlen : int, optional
        Maximum length, or 0 for inifinite. By default 0.
    """        
    def __init__(self, maxlen: int = 0):
        if maxlen < 0:
            raise ValueError(f'max length should be positive or zero, got {maxlen}')
        
        self.maxlen = maxlen
        self._data = list()   

        self._min_unvisited = 0

    def insert(self, priority : P, value : V, trim: bool = True, prevent_duplicates: bool = False) -> None:
        """
        Insert new unvisited item to queue, optionally preventing duplicated entries.
        When inserting multiple items at a time, can postpone trimming to max length.

        Entry is considered duplicated if it has the same value *and* priority.

        Parameters
        ----------
        priority : P
            Insert at this priority.        
        value : V
            Item to insert.
        trim : bool, optional
            By default True. Set to false if inserting many items at at time, and call `self.trim()`. 
        prevent_duplicates : bool, optional
            Set `True` to allow duplicated items in the queue. Default is `False`.
        """        
        # Find insertion pointf
        wrapped = PriorityWrapper(priority, value, False)
        i = bisect_left(self._data, wrapped)
        # If trying to insert capacity, nothing to do
        if self.maxlen > 0 and i >= self.maxlen:
            return
        # If the item is already in the queue, nothing to do
        if prevent_duplicates and i < len(self._data):
            if self._data[i].value == value:
                return
        # Insert
        self._data.insert(i, wrapped)
        # Keep track of highest priority visited item
        self._min_unvisited = min(self._min_unvisited, i)
       
        if trim:
            self.trim()
        
    def trim(self) -> None:
        """
        Trim queue to maximum length. Used with `insert(x, trim=False)`.
        """    
        if self.maxlen == 0:
            return    
        self._data = self._data[:self.maxlen]
        assert self._min_unvisited <= self.maxlen

    def size(self) -> int:
        """Return number of items in queue (both visited and unvisited)."""        
        return len(self._data)

    def visit(self) -> Generator[V]:
        """
        Iterate over unvisited items from lowest to highest.
        Unlike most iterators, this one allows insertions to the queue while iterating.

        Yields
        ------
        out : V
            Next smallest unvisited item.
        """        
        while self.has_unvisited():
            yield self.next_smallest_unvisited()

    def has_unvisited(self) -> bool:
        """
        Returns True if there are unvisited items in queue else False.
        """
        return self._min_unvisited < len(self._data)

    def next_smallest_unvisited(self) -> V:
        """
        Returns smallest unvisited item in queue.
        Will cause error if there are no more such items, so call `has_visited` before.
        """
        self._data[self._min_unvisited].visited = True
        x = self._data[self._min_unvisited].value
        while self._min_unvisited < len(self._data) and self._data[self._min_unvisited].visited:
            self._min_unvisited += 1
        return x
    
    def ksmallest(self, k: int = 1) -> list[V]:
        """
        Return k smallest item in queue (or less, if queue is not large enough).

        Parameters
        ----------
        k : int, optional
            How many items to return, by default 1

        Returns
        -------
        list[T]
            List of k smallest items (or fewer if k > size of queue).
        """        
        return [ item.value for item in self._data[:k] ]


if __name__ == '__main__':
    print(f'Running quick-and-dirty unit tests in {__file__}')

    import random
    from collections import deque

    random.seed(f'random string for {__file__}')

    visited = set()
    def validate(q, expected_order, expected_size):
        i = 0
        for x in q.visit():
            assert x == expected_order[i]
            assert x not in visited
            visited.add(x)
            i += 1
        assert i == len(expected_order)
        assert q.size() == expected_size

    q = VisitPriorityQueue(3)

    # Basic insert with limited size
    # Should only see a, b, c
    q.insert(9, 'f')
    q.insert(5, 'b')
    q.insert(4, 'a')
    q.insert(6, 'c')
    validate(q, ['a','b','c'], 3)
    assert q.ksmallest(2) ==  ['a', 'b'] 
    assert q.ksmallest(3) ==  ['a', 'b', 'c'] 
    assert q.ksmallest(100) ==  q.ksmallest(3)

    # Test that visited are still in queue, later inserts cannot go in beyond queue size
    # Should only see a, b, c
    q.insert(9,'d')
    q.insert(10,'b')
    q.insert(11,'a')
    validate(q, [], 3) # a,b,c already visited, but are still in queue
    assert q.ksmallest(3) ==  ['a', 'b', 'c'] 

    # Insert into middle of queue, should push c out.
    q.insert(2,'Y')
    assert q.ksmallest(3) ==  ['Y', 'a', 'b'] 
    assert q.ksmallest(100) ==  q.ksmallest(3)
    assert q.ksmallest(1) ==  ['Y'] 
    validate(q, ['Y'], 3) # Y is the only one not visited
    
    # Untrimmed inserts into beginning, middle, and end of queue -- followed by trim.
    # Should push a and b out
    q.insert(1, 'X', False)
    q.insert(7, 'd', False)
    q.insert(3, 'Z', False)
    q.insert(8, 'e', False)
    q.trim()
    validate(q, ['X','Z'], 3) # Y already visited, 'X' and 'Z' are not
    assert q.ksmallest(2) ==  ['X', 'Y'] 
    assert q.ksmallest(3) ==  ['X', 'Y', 'Z'] 
    assert q.ksmallest(100) ==  q.ksmallest(3)

    # Test duplicate detection during iteration
    q.insert(2,'Y', prevent_duplicates=True)
    validate(q, [], 3) # Y already visited and was not reinserted
    assert q.ksmallest(2) ==  ['X', 'Y'] 
    assert q.ksmallest(3) ==  ['X', 'Y', 'Z'] 

    # Test insertion during iteration
    n = 10000
    xs = list(range(n))
    random.shuffle(xs)
    xs = deque(xs)

    visited.clear()    
    q = VisitPriorityQueue(1000)
    inserted = set()
    
    # Insert 1 item
    p = xs.popleft()
    q.insert(p, str(p))
    inserted.add(p)

    # Insert 10 items in each iteration.
    for x in q.visit():
        p = int(x)
        assert p == min(inserted)
        assert p not in visited        
        visited.add(p)
        inserted.remove(p)
        for i in range(min(10, len(xs))):
            np = xs.popleft()
            q.insert(np, str(np))
            inserted.add(np)                        
    # We should have used up all of the xs
    assert len(xs) == 0
    # q will end up storing the smallest values in xs
    assert q.ksmallest(1000) == [ str(i) for i in range(1000) ]
    
    print('Done.')