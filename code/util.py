import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from torch.utils.data import _DatasetKind
from torch.utils.data import _utils

class MyCustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor=2,
                 persistent_workers=False, pin_memory_device="",
                 replace_size=3):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                         num_workers, collate_fn, pin_memory, drop_last,
                         timeout, worker_init_fn, multiprocessing_context,
                         generator, prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)
        self.replace_size = replace_size

    def _get_iterator(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

class _SingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        self._replace_size = loader.replace_size
        self._replaced_indices = []
        super().__init__(loader)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration

        if len(self._replaced_indices) >= self._replace_size:
            replace_indices = self._replaced_indices[:self._replace_size]
            if self._auto_collation:
                data = [self._dataset_fetcher.fetch([idx]) for idx in replace_indices]
            else:
                data = [self._dataset_fetcher.fetch(idx) for idx in replace_indices]
            self._replaced_indices = self._replaced_indices[self._replace_size:]
        else:
            data = self._dataset_fetcher.fetch(index)

        self._replaced_indices.append(index)

        if self._pin_memory:
            data = torch.utils.data._utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data

'''
class _MultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self._replace_size = loader.replace_size
        self._replaced_indices = []
        super().__init__(loader)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0
        self._rcvd_idx = 0
        self._task_info = {}
        self._tasks_outstanding = 0
        self._replaced_indices = []

        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_put_index(self):
        if self._tasks_outstanding < self._prefetch_factor * self._num_workers:
            if len(self._replaced_indices) >= self._replace_size:
                replace_indices = self._replaced_indices[:self._replace_size]
                for idx in replace_indices:
                    for _ in range(self._num_workers):  # find the next active worker, if any
                        worker_queue_idx = next(self._worker_queue_idx_cycle)
                        if self._workers_status[worker_queue_idx]:
                            break
                    else:
                        continue

                    self._index_queues[worker_queue_idx].put((self._send_idx, idx))  # Pass index directly
                    self._task_info[self._send_idx] = (worker_queue_idx,)
                    self._tasks_outstanding += 1
                    self._send_idx += 1

                self._replaced_indices = self._replaced_indices[self._replace_size:]
            else:
                try:
                    index = self._next_index()
                except StopIteration:
                    return
                for _ in range(self._num_workers):  # find the next active worker, if any
                    worker_queue_idx = next(self._worker_queue_idx_cycle)
                    if self._workers_status[worker_queue_idx]:
                        break
                else:
                    return

                self._index_queues[worker_queue_idx].put((self._send_idx, index))  # Pass index directly
                self._task_info[self._send_idx] = (worker_queue_idx,)
                self._tasks_outstanding += 1
                self._send_idx += 1

                self._replaced_indices.append(index)

    def _next_data(self):
        while True:
            if self._rcvd_idx == self._send_idx:
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            info = self._task_info.pop(self._rcvd_idx)
            worker_id = info[0]
            if len(info) == 2:
                data = info[1]
                return self._process_data(data)

            if self._workers_status[worker_id]:
                idx, data = self._get_data()
                self._tasks_outstanding -= 1
                if idx != self._rcvd_idx:
                    self._task_info[idx] = (worker_id, data)
                else:
                    return self._process_data(data)

            self._rcvd_idx += 1
'''


class _MultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self._replace_size = loader.replace_size
        self._current_batch = []
        self._batch_size = loader.batch_size  # Store the batch size in the iterator
        super().__init__(loader)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0
        self._rcvd_idx = 0
        self._task_info = {}
        self._tasks_outstanding = 0
        self._current_batch = []

        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_put_index(self):
        if self._tasks_outstanding < self._prefetch_factor * self._num_workers:
            if len(self._current_batch) >= self._batch_size:
                # If we have a full batch, replace the specified number of samples
                replace_indices = self._current_batch[:self._batch_size - self._replace_size]
                self._current_batch = self._current_batch[self._batch_size - self._replace_size:]
                for idx in replace_indices:
                    for _ in range(self._num_workers):  # find the next active worker, if any
                        worker_queue_idx = next(self._worker_queue_idx_cycle)
                        if self._workers_status[worker_queue_idx]:
                            break
                    else:
                        continue

                    self._index_queues[worker_queue_idx].put((self._send_idx, idx))
                    self._task_info[self._send_idx] = (worker_queue_idx,)
                    self._tasks_outstanding += 1
                    self._send_idx += 1
            else:
                # If we don't have a full batch, add new samples
                try:
                    index = self._next_index()
                except StopIteration:
                    return
                for _ in range(self._num_workers):  # find the next active worker, if any
                    worker_queue_idx = next(self._worker_queue_idx_cycle)
                    if self._workers_status[worker_queue_idx]:
                        break
                else:
                    return

                self._index_queues[worker_queue_idx].put((self._send_idx, index))
                self._task_info[self._send_idx] = (worker_queue_idx,)
                self._tasks_outstanding += 1
                self._send_idx += 1

                self._current_batch.append(index)

    def _next_data(self):
        while True:
            if self._rcvd_idx == self._send_idx:
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            info = self._task_info.pop(self._rcvd_idx)
            worker_id = info[0]
            if len(info) == 2:
                data = info[1]
                return self._process_data(data)

            if self._workers_status[worker_id]:
                idx, data = self._get_data()
                self._tasks_outstanding -= 1
                if idx != self._rcvd_idx:
                    self._task_info[idx] = (worker_id, data)
                else:
                    return self._process_data(data)

            self._rcvd_idx += 1