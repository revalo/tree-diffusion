# import multiprocessing as mp
from pathos.helpers import mp
import multiprocessing.managers
from abc import ABC, abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
from typing import TypeVar

import numpy as np
import tree  # noqa

T = TypeVar("T")
NestedArray = tree.StructureKV[str, np.ndarray]


class ArrayView:
    def __init__(
        self,
        multiprocessing_array: mp.Array,
        numpy_array: np.ndarray,
        num_items: int,
    ):
        self.num_items = num_items
        self.dtype = numpy_array.dtype
        self.shape = (num_items, *numpy_array.shape)
        self.nbytes: int = numpy_array.nbytes * num_items

        self._item_shape = numpy_array.shape
        self._multiprocessing_array = multiprocessing_array

        self._array_view = np.frombuffer(
            buffer=multiprocessing_array,
            dtype=numpy_array.dtype,
            count=np.product(self.shape),
        ).reshape(self.shape)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        if "_view" in state:
            del self.__dict__["_view"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._array_view = np.frombuffer(
            buffer=self._multiprocessing_array,
            dtype=self.dtype,
            count=np.product(self.shape),
        ).reshape(self.shape)

    def put(self, item: np.ndarray, index: int) -> None:
        assert item.shape == self._item_shape and item.dtype == self.dtype
        self._array_view[index, ...] = item

    def get(self, index: int) -> np.ndarray:
        return np.copy(self._array_view[index, ...])


class AbstractQueue(ABC):
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._queue = mp.Queue(maxsize=maxsize)

    @abstractmethod
    def put(self, item: T) -> None:
        # https://stackoverflow.com/a/42778801
        raise NotImplementedError

    @abstractmethod
    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        raise NotImplementedError

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def qsize(self) -> int:
        return self._queue.qsize()


class ArrayQueue(AbstractQueue):
    def __init__(self, array: np.ndarray, maxsize: int):
        super(ArrayQueue, self).__init__(maxsize=maxsize)

        self._lock = mp.Lock()
        self._next_index = mp.Value("i", 0)
        self.nbytes: int = array.nbytes * maxsize

        self._array = mp.Array("c", self.nbytes)
        self._array_view = ArrayView(
            multiprocessing_array=self._array.get_obj(),
            numpy_array=array,
            num_items=maxsize,
        )

    def put(self, array: np.ndarray) -> None:
        # Avoid several simultaneous 'put' call
        with self._next_index.get_lock():
            self._queue.put(self._next_index.value)
            # Avoid ArrayQueue changes during a 'put' or 'get' call
            with self._lock:
                self._put(array=array, index=self._next_index.value)
            self._next_index.value = (self._next_index.value + 1) % self.maxsize

    def _put(self, array: np.ndarray, index: int) -> None:
        self._array_view.put(array, index)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> np.ndarray:
        index = self._queue.get(block=block, timeout=timeout)
        # Avoid ArrayQueue changes during a 'put' or 'get' call
        with self._lock:
            return self._get(index=index)

    def _get(self, index: int) -> np.ndarray:
        return self._array_view.get(index=index)


class SimpleTreeQueue(AbstractQueue):
    """TreeQueue implemented with simple locking techniques."""

    def __init__(self, nested_array: NestedArray, maxsize: int):
        super().__init__(maxsize=maxsize)

        self._lock = mp.Lock()
        self._next_index = mp.Value("i", 0)

        self._nested_queue = tree.map_structure(
            lambda array: ArrayQueue(array=array, maxsize=maxsize), nested_array
        )

        self._nested_array = nested_array
        self.nbytes = sum([q.nbytes for q in tree.flatten(self._nested_queue)])

    def put(
        self,
        nested_array: NestedArray,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        # Avoid several simultaneous 'put' call
        with self._next_index.get_lock():
            self._queue.put(self._next_index.value, block=block, timeout=timeout)
            # Avoid ArrayQueue changes during a 'put' or 'get' call
            with self._lock:
                tree.map_structure(
                    lambda queue, array: queue._put(  # noqa
                        array=array, index=self._next_index.value
                    ),
                    self._nested_queue,
                    nested_array,
                )
            self._next_index.value = (self._next_index.value + 1) % self.maxsize

    def get(self, block: bool = True, timeout: Optional[float] = None) -> NestedArray:
        index = self._queue.get(block=block, timeout=timeout)
        # Avoid ArrayQueue changes during a 'put' or 'get' call
        with self._lock:
            return tree.map_structure(
                lambda queue: queue._get(index=index),  # noqa
                self._nested_queue,
            )


class TreeQueue(AbstractQueue):
    """TreeQueue implemented with techniques allowing it to be more efficient when using many
    simultaneous processes and threads.
    """

    def __init__(self, nested_array: NestedArray, maxsize: int):
        super().__init__(maxsize=maxsize)

        self._get_lock = mp.Lock()
        self._put_lock = mp.Lock()
        self._condition = mp.Condition()
        self._next_index = mp.Value("i", 0)

        self._manager = mp.Manager()
        self._active_get_index_dict = self._manager.dict()
        self._active_put_index_dict = self._manager.dict()

        self._nested_queue = tree.map_structure(
            lambda array: ArrayQueue(array=array, maxsize=maxsize), nested_array
        )

        self._nested_array = nested_array
        self.nbytes = sum([q.nbytes for q in tree.flatten(self._nested_queue)])

    def put(self, nested_array: NestedArray, block=True, timeout=None) -> None:
        with self._put_lock:
            index = self._next_index.value
            self._next_index.value = (index + 1) % self.maxsize
            self.wait_and_add(index, self._active_put_index_dict, self._condition)

        while index in self._active_get_index_dict.keys():
            with self._condition:
                self._condition.wait()

        tree.map_structure(
            lambda queue, array: queue._put(  # noqa
                array=array, index=index
            ),
            self._nested_queue,
            nested_array,
        )

        # put only in queue the index after being sure that the nested_array is written in the nested queue
        self._queue.put(index, block=block, timeout=timeout)
        del self._active_put_index_dict[index]
        with self._condition:
            self._condition.notify_all()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> NestedArray:
        with self._get_lock:
            index = self._queue.get(block=block, timeout=timeout)
            # The index in the queue are always index of element that are finished to be transferred,
            # we therefore don't need to enquiry if the index is in the put dict.
            self.wait_and_add(index, self._active_get_index_dict, self._condition)

        nested_array = tree.map_structure(
            lambda queue: queue._get(index=index),  # noqa
            self._nested_queue,
        )

        del self._active_get_index_dict[index]
        with self._condition:
            self._condition.notify_all()
        return nested_array

    @staticmethod
    def wait_and_add(
        index: int,
        dictionary: Dict[int, bool],
        condition: multiprocessing.Condition,
    ) -> None:
        """The following code make will acquire lock, test if it is in the get dictionary, then,
        it'll either add it if it is not, else will wait for the next notify.

        This is an equivalent to multiprocessing.Condition().wait_for() except that the lock is held
        to do an action, in that case adding an entry to the dictionary.

        References for the code:
        Acquire lock inside a try-finally block:
        https://stackoverflow.com/a/14137638
        While loop with condition until predicate is met (and snippet condition with-wait):
        https://stackoverflow.com/a/23116848
        Equivalent while True and while loop:
        https://stackoverflow.com/a/27512815
        multiprocessing.Condition().wait_for():
        https://docs.python.org/3/library/threading.html#threading.Condition.wait_for

        Args:
            index: index that will be tested for its presence in the dict before being added
            dictionary: multiprocessing dict
            condition: multiprocessing condition
        """

        while True:
            condition.acquire()
            try:
                if index not in dictionary.keys():
                    dictionary[index] = True
                    break
                condition.wait()
            finally:
                condition.release()
