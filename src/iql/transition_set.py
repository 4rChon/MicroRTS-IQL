import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np

TransitionSetSample = tuple[
    np.ndarray[Any, np.dtype[np.int32]],  # states
    np.ndarray[Any, np.dtype[np.int32]],  # actions
    np.ndarray[Any, np.dtype[np.int32]],  # action_masks
    np.ndarray[Any, np.dtype[np.float32]],  # rewards
    np.ndarray[Any, np.dtype[np.int32]],  # next_states
    np.ndarray[Any, np.dtype[np.bool_]],    # dones
]


class TransitionSet():
    def __init__(
        self,
        path: Path,
        map_size_gb: int = 25,
    ):
        self.metadata = None

        self._lmdb_path = path
        self._map_size_gb = map_size_gb
        self._chunk_pointer = -1
        self._idx_pointer = 0
        self._states = None
        self._actions = None
        self._action_masks = None
        self._rewards = None
        self._next_states = None
        self._dones = None
        self._worker_id = 0
        self._lmdb_env = None

    # chunk_size getter
    @property
    def chunk_size(self) -> int:
        if self.metadata is None:
            return 0
        return self.metadata["chunk_size"]

    @property
    def size(self) -> int:
        if self.metadata is None:
            return -1
        return self.metadata["size"]

    @property
    def chunk_count(self) -> int:
        return self.__len__() // self.chunk_size

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, '_lmdb_env') or self._lmdb_env is None:
            return self._open_lmdb()
        return self._lmdb_env

    def init_lmdb(self, worker_id: int = 0) -> None:
        print("Initializing LMDB environment for worker", worker_id)
        self._open_lmdb()
        self._init_metrics()

        self._worker_id = worker_id

    def _open_lmdb(self) -> lmdb.Environment:
        self._lmdb_env = lmdb.open(
            str(self._lmdb_path),
            map_size=self._map_size_gb * 1024 ** 3,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            metasync=False,
            sync=False,
        )

        return self._lmdb_env

    def _init_metrics(self) -> None:
        with self.lmdb_env.begin(write=False, buffers=True) as txn:
            data = txn.get("metadata".encode("ascii"))
            if data is None:
                raise ValueError(
                    "LMDB metadata not found. Please ensure the LMDB "
                    "environment is properly initialized."
                )
            self.metadata = pickle.loads(data)  # type: ignore
            print(f"LMDB metadata: {self.metadata}")

    def _load_chunk(self, chunk_idx: int) -> None:
        with self.lmdb_env.begin(write=False, buffers=True) as txn:
            print(
                f"Worker ID: {self._worker_id} loading chunk {chunk_idx} from\
                lmdb"
            )
            data = txn.get(f"{chunk_idx}".encode("ascii"))
            if data is None:
                raise ValueError(
                    f"Chunk {chunk_idx} not found in LMDB. Please ensure "
                    "the chunk exists."
                )
            (
                self._states,
                self._actions,
                self._action_masks,
                self._rewards,
                self._next_states,
                self._dones
            ) = pickle.loads(data)  # type: ignore

    def __len__(self):
        if self.size == -1 and self._lmdb_env is None:
            self._open_lmdb()
            self._init_metrics()
            self._lmdb_env = None

        return self.size

    def sample_next(
        self, batch_size: int, reward_scale: float = 1.0
    ) -> TransitionSetSample:
        chunk_idx = (self._idx_pointer // self.chunk_size) % self.chunk_count
        chunk_idx_within = self._idx_pointer % self.chunk_size
        from_idx = chunk_idx_within
        to_idx = min(chunk_idx_within + batch_size, self.chunk_size)

        if to_idx - from_idx < batch_size:
            self._idx_pointer += batch_size - (to_idx - from_idx)
            self._idx_pointer %= self.__len__()
            return self.sample_next(batch_size)

        if chunk_idx != self._chunk_pointer:
            self._load_chunk(chunk_idx)
            self._chunk_pointer = chunk_idx % self.chunk_count

        self._idx_pointer += batch_size
        self._idx_pointer %= self.__len__()
        return (
            self._states[from_idx:to_idx],  # type: ignore
            self._actions[from_idx:to_idx],  # type: ignore
            self._action_masks[from_idx:to_idx],  # type: ignore
            self._rewards[from_idx:to_idx] * reward_scale,  # type: ignore
            self._next_states[from_idx:to_idx],  # type: ignore
            self._dones[from_idx:to_idx]  # type: ignore
        )

    def close(self):
        self.lmdb_env.close()
