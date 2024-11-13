from bisect import bisect_left
from collections.abc import Callable, Generator
from typing import TypeVar, Any
from itertools import islice, filterfalse
from dataclasses import dataclass

T = TypeVar('T')

@dataclass(order=True)
class VisitWrapper():
    """Keep Track of visited items in queue."""
    x: T
    visited: bool = False

class VisitPriorityQueue():
    """
    A variant of priority queue with two crucial differences:
    it supports limited size, and it can filter returns to previously unvisited elements.

    Parameters
    ----------
    maxlen : int, optional
        Maximum length, or 0 for inifinite. By default 0.
    key : Callable[[T], Any] | None, optional
        Key used for comparisons, by default None.
    """        
    def __init__(self, maxlen: int = 0, 
                 key: Callable[[T], Any] | None = None):
        if maxlen < 0:
            raise ValueError(f'max length should be positive or zero, got {maxlen}')
        
        self.maxlen = maxlen
        self._data = list()                

        # Since we are wrapping objects, need to unwrap them when using a key
        if key is not None:
            self._key = lambda wrapped: key(wrapped.x)
        else:
            self._key = None

        self._min_unvisited = 0

    def insert(self, x : T, trim: bool = True) -> None:
        """
        Insert new unvisited item to queue.
        When inserting multiple items at a time, can postpone trimming to max length.

        Parameters
        ----------
        x : T
            Item to insert.
        trim : bool, optional
            By default True. Set to false if inserting many items at at time, and call `self.trim()`. 
        """        
        # Find insertion point
        wrapped = VisitWrapper(x, False)
        i = bisect_left(self._data, self._key(wrapped) if self._key is not None else wrapped, key=self._key)
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
        self._data = self._data[:self.maxlen]
        assert self._min_unvisited <= self.maxlen

    def size(self) -> int:
        """Return number of items in queue (both visited and unvisited)."""        
        return len(self._data)

    def visit(self) -> Generator[T]:
        """
        Iterate over unvisited items from lowest to highest.
        Unlike most iterators, this one allows insertions to the queue while iterating.

        Yields
        ------
        out : T
            Next smallest unvisited item.
        """        
        while self.has_unvisited():
            yield self.next_smallest_unvisited()

    def has_unvisited(self) -> bool:
        """
        Returns True if there are unvisited items in queue else False.
        """
        return self._min_unvisited < len(self._data)

    def next_smallest_unvisited(self) -> T:
        """
        Returns smallest unvisited item in queue.
        Will cause error if there are no more such items, so call `has_visited` before.
        """
        self._data[self._min_unvisited].visited = True
        x = self._data[self._min_unvisited].x        
        while self._min_unvisited < len(self._data) and self._data[self._min_unvisited].visited:
            self._min_unvisited += 1
        return x
    
    def ksmallest(self, k: int = 1) -> list[T]:
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
        return [ item.x for item in self._data[:k] ]


if __name__ == '__main__':
    print(f'Running quick-and-dirty unit tests in {__file__}')

    import random
    from collections import deque

    random.seed(f'random string for {__file__}')

    visited = set()
    def validate(q, expected_order, expected_size):
        i = 0
        for p, x in q.visit():
            assert x == expected_order[i]
            assert x not in visited
            visited.add(x)
            i += 1
        assert i == len(expected_order)
        assert q.size() == expected_size

    q = VisitPriorityQueue(3)

    # Basic insert with limited size
    # Should only see a, b, c
    q.insert((9, 'f'))
    q.insert((5, 'b'))
    q.insert((4, 'a'))
    q.insert((6, 'c'))
    validate(q, ['a','b','c'], 3)
    assert q.ksmallest(2) ==  [(4, 'a'), (5, 'b')] 
    assert q.ksmallest(3) ==  [(4, 'a'), (5, 'b'), (6, 'c')] 
    assert q.ksmallest(100) ==  q.ksmallest(3)

    # Test that visited are still in queue, later inserts cannot go in beyond queue size
    # Should only see a, b, c
    q.insert((9,'d'))
    q.insert((10,'b'))
    q.insert((11,'a'))
    validate(q, [], 3) # a,b,c already visited, but are still in queue
    assert q.ksmallest(3) ==  [(4, 'a'), (5, 'b'), (6, 'c')] 

    # Insert into middle of queue, sould push c out.
    q.insert((2,'Y'))
    assert q.ksmallest(3) ==  [(2,'Y'), (4, 'a'), (5, 'b')] 
    assert q.ksmallest(100) ==  q.ksmallest(3)
    assert q.ksmallest(1) ==  [(2,'Y')] 
    validate(q, ['Y'], 3) # Y is the only one not visited
    
    # Untrimmed inserts into beginning, middle, and end of queue -- followed by trim.
    # Should push a and b out
    q.insert((1, 'X'), False)
    q.insert((7, 'd'), False)
    q.insert((3, 'Z'), False)
    q.insert((8, 'e'), False)
    q.trim()
    validate(q, ['X','Z'], 3) # Y already visited, 'X' and 'Z' are not
    assert q.ksmallest(2) ==  [(1, 'X'), (2, 'Y')] 
    assert q.ksmallest(3) ==  [(1, 'X'), (2, 'Y'), (3, 'Z')] 
    assert q.ksmallest(100) ==  q.ksmallest(3)

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
    q.insert((p, str(p)))
    inserted.add(p)

    # Insert 10 items in each iteration.
    for p, x in q.visit():
        assert p == min(inserted)
        assert p not in visited        
        visited.add(p)
        inserted.remove(p)
        for i in range(min(10, len(xs))):
            np = xs.popleft()
            q.insert((np, str(np)))
            inserted.add(np)                        
    # We should have used up all of the xs
    assert len(xs) == 0
    # q will end up storing the smallest values in xs
    assert q.ksmallest(1000) == [ (i, str(i)) for i in range(1000) ]
    




        

    

    

