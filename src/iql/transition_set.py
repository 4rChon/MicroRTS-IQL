import pickle
from pathlib import Path

import lmdb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

TransitionSetSample = tuple[
    torch.Tensor,  # states
    torch.Tensor,  # next states
    torch.Tensor,  # actions
    # torch.Tensor,  # next_actions
    torch.Tensor,  # rewards
    torch.Tensor,  # dones
    torch.Tensor,  # action_masks
    # torch.Tensor,  # next_action_masks
]


PACKED_ACTION_DIMS = [6, 10, 14, 18, 22, 29]
STATE_DIMS = [5, 5, 3, 8, 6, 2]
ACTION_DIMS = [6, 4, 4, 4, 4, 7, 49]


def vectorised_packed_state_to_obs(
    packed_state: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    one_hots = []
    for i, dim in enumerate(STATE_DIMS):
        oh = F.one_hot(packed_state[:, i], num_classes=dim)
        one_hots.append(oh)

    cat_oh = torch.cat(one_hots, dim=-1)
    return cat_oh.float().view(H, W, -1)


def vectorised_packed_action_to_action(
    packed_action: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    action = torch.zeros((
        H * W, sum(ACTION_DIMS)
    ), dtype=torch.long)
    for i, grid_action in enumerate(packed_action):
        action_type = grid_action[0]
        if action_type == 0:
            continue

        action[i][action_type] = 1
        if action_type >= 4:
            action[i][
                PACKED_ACTION_DIMS[action_type] +
                grid_action[action_type + 1]
            ] = 1

        if action_type != 5:
            action[i][
                PACKED_ACTION_DIMS[action_type - 1] +
                grid_action[action_type]
            ] = 1

    return action.float().reshape(
        H, W, sum(ACTION_DIMS)
    )


class TransitionSet(Dataset):
    def __init__(
        self,
        path: Path,
        state_dim: tuple[int, int, int],
        action_dim: list[int],
        map_size_gb: int = 25,
        reward_scale: float = 1.0
    ):
        print("Initializing TransitionSet with path:", path)
        self.metadata = None

        self._lmdb_path = path
        self._map_size_gb = map_size_gb
        self._worker_id = 0
        self._lmdb_env = None
        self._reward_scale: float = reward_scale
        self._state_dim = state_dim
        self._action_dim = action_dim
        self.length = -1

    @property
    def size(self) -> int:
        return self.length

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, '_lmdb_env') or self._lmdb_env is None:
            return self._open_lmdb()
        return self._lmdb_env

    def __len__(self):
        if self.size == -1 and self._lmdb_env is None:
            self._open_lmdb()
            self._init_metrics()
            self._lmdb_env = None

        return self.size

    def __getitem__(self, idx: int) -> TransitionSetSample:
        with self.lmdb_env.begin(write=False, buffers=True) as txn:
            data = txn.get(f"{idx}".encode("ascii"))
            if data is None:
                raise ValueError(
                    f"Idx {idx} not found in LMDB. Please ensure "
                    "the key exists."
                )
            transition = pickle.loads(data)  # type: ignore

            h, w, _ = self._state_dim
            act_c = sum(self._action_dim)

            transition = [
                torch.tensor(b)
                for b in transition
            ]
            s, ns, a, _, r, d, am, _ = transition
            am = am.reshape(h, w, act_c).float()
            s = vectorised_packed_state_to_obs(s.long(), h, w)
            ns = vectorised_packed_state_to_obs(ns.long(), h, w)
            a = vectorised_packed_action_to_action(a.long(), h, w)

            return (
                s,                               # states
                ns,                              # next states
                a,                               # actions
                r.float() * self._reward_scale,  # rewards
                d.float(),                       # dones
                am,                              # action masks
            )

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

        if self._lmdb_env is None:
            raise ValueError("Failed to open LMDB environment.")
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

            self.length = self.metadata["num_transitions"]  # type: ignore

    def init_lmdb(self, worker_id: int = 0) -> None:
        print("Initializing LMDB environment for worker", worker_id)
        self._open_lmdb()
        self._init_metrics()

        self._worker_id = worker_id

    def close(self):
        self.lmdb_env.close()


class TransitionDataLoader(DataLoader):
    def __init__(
        self,
        transition_set: TransitionSet,
        batch_size: int,
        num_workers: int,
        num_samples: int
    ):
        num_samples *= batch_size * num_workers
        print(f"Creating TransitionDataLoader with batch size: {batch_size} \
              and num workers: {num_workers}")
        self.transition_set = transition_set
        sampler = RandomSampler(
            transition_set, replacement=True, num_samples=num_samples
        )
        super().__init__(
            dataset=transition_set,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            worker_init_fn=transition_set.init_lmdb,
            pin_memory=True,
            sampler=sampler,
            timeout=30
        )
