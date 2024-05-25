from pathos.helpers import mp

from .treequeues import TreeQueue


class ParallelDataLoader(object):
    def __init__(
        self, make_batch_generator, num_workers=mp.cpu_count(), queue_size_multiplier=8
    ):
        self._make_batch_generator = make_batch_generator
        self._num_workers = num_workers
        self._data_queue = mp.Queue(maxsize=num_workers * queue_size_multiplier)

        def _worker_fn(make_batch_fn, data_queue):
            batch_fn = make_batch_fn()
            while True:
                batch = batch_fn()
                data_queue.put(batch)

        self._processes = [
            mp.Process(
                target=_worker_fn,
                args=(self._make_batch_generator, self._data_queue),
                daemon=True,
            )
            for _ in range(self._num_workers)
        ]

    def start(self):
        for p in self._processes:
            p.start()

    def stop(self):
        for p in self._processes:
            p.terminate()
            p.join()

    def get(self):
        return self._data_queue.get()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class TreeQueueParallelDataLoader(object):
    def __init__(
        self, generate_batch_fn, num_workers=mp.cpu_count(), queue_size_multiplier=8
    ):
        self._generate_batch_fn = generate_batch_fn
        self._num_workers = num_workers
        self._sample_result = generate_batch_fn()
        self._sample_result_kv = {f"{i}": x for i, x in enumerate(self._sample_result)}
        self._data_queue = TreeQueue(
            self._sample_result_kv, maxsize=num_workers * queue_size_multiplier
        )

        def _worker_fn(batch_fn, data_queue):
            while True:
                batch = batch_fn()
                data_queue.put({f"{i}": x for i, x in enumerate(batch)})

        self._processes = [
            mp.Process(
                target=_worker_fn,
                args=(self._generate_batch_fn, self._data_queue),
                daemon=True,
            )
            for _ in range(self._num_workers)
        ]

    def start(self):
        for p in self._processes:
            p.start()

    def stop(self):
        for p in self._processes:
            p.terminate()
            p.join()

    def get(self):
        rv = self._data_queue.get()
        return tuple(rv[k] for k in sorted(rv.keys()))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
