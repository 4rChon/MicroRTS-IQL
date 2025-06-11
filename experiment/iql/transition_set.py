import pickle
import lmdb

class TransitionSet():
    def __init__(
        self,
        path: str,
        map_size_gb: int = 25,
    ):
        self.metadata = None

        self._lmdb_path = path
        self._map_size_gb = map_size_gb
        self._lmdb_env = None
        self._chunk_pointer = -1
        self._idx_pointer = 0
        self._states = None
        self._actions = None
        self._action_masks = None
        self._rewards = None
        self._next_states = None
        self._dones = None
        self._worker_id = 0

    # chunk_size getter
    @property
    def chunk_size(self):
        if self.metadata is None:
            return 0
        return self.metadata["chunk_size"]
    
    # size getter: use this like so: 
    @property
    def size(self):
        if self.metadata is None:
            return -1
        return self.metadata["size"]
    
    @property
    def chunk_count(self):
        return self.__len__() // self.chunk_size
    
    def init_lmdb(self, worker_id: int = 0):
        print("Initializing LMDB environment for worker", worker_id)
        self._open_lmdb()
        self._init_metrics()

        self._worker_id = worker_id

    def _open_lmdb(self):
        self._lmdb_env = lmdb.open(
            self._lmdb_path,
            map_size=self._map_size_gb * 1024 ** 3,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            metasync=False,
            sync=False,
        )

    def _init_metrics(self):
        with self._lmdb_env.begin(write=False, buffers=True) as txn:
            self.metadata = pickle.loads(txn.get("metadata".encode("ascii")))
            print(f"LMDB metadata: {self.metadata}")

    def _load_chunk(self, chunk_idx: int):
        with self._lmdb_env.begin(write=False, buffers=True) as txn:
            print(f"Worker ID: {self._worker_id} loading chunk {chunk_idx} from lmdb")
            data = txn.get(f"{chunk_idx}".encode("ascii"))
            self._states, self._actions, self._action_masks, self._rewards, self._next_states, self._dones = pickle.loads(data)

    def __len__(self):
        if self.size == -1 and self._lmdb_env is None:
            self._open_lmdb()
            self._init_metrics()
            self._lmdb_env = None

        return self.size

    def sample_next(self, batch_size: int, reward_scale: float = 1.0):
        chunk_idx = (((self._idx_pointer // self.chunk_size) * 12) + (self._worker_id)) % self.chunk_count
        chunk_idx_within = self._idx_pointer % self.chunk_size
        from_idx = chunk_idx_within
        to_idx = min(chunk_idx_within + batch_size, self.chunk_size)

        if to_idx - from_idx < batch_size:
            self._idx_pointer += batch_size - (to_idx - from_idx)
            self._idx_pointer %= self.__len__() # wrap around to the beginning to avoid overflow
            return self.sample_next(batch_size)
        
        if chunk_idx != self._chunk_pointer:
            if self._lmdb_env is None:
                self.init_lmdb()

            self._load_chunk(chunk_idx)
            self._chunk_pointer = chunk_idx % self.chunk_count

        self._idx_pointer += batch_size
        self._idx_pointer %= self.__len__() # wrap around to the beginning to avoid overflow
        return self._states[from_idx:to_idx], \
            self._actions[from_idx:to_idx], \
            self._action_masks[from_idx:to_idx], \
            self._rewards[from_idx:to_idx] * reward_scale, \
            self._next_states[from_idx:to_idx], \
            self._dones[from_idx:to_idx]
        
    def close(self):
        self._lmdb_env.close()