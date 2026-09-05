# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License

import numpy as np

from math import hypot
from typing import List, Tuple

from pyppbox.utils.persontools import Person
from pyppbox.utils.logtools import add_error_log, add_warning_log
from scipy.optimize import linear_sum_assignment

try:
    import lap
except Exception:
    lap = None


def _linear_assignment(cost_matrix: np.ndarray):
    """Return one-to-one assignments, falling back to SciPy if LAP is unavailable."""
    cm = np.ascontiguousarray(cost_matrix, dtype=np.float32)
    if lap is not None:
        try:
            return lap.lapjvxa(cm, extend_cost=cm.shape[0] != cm.shape[1], return_cost=False)
        except Exception as e:
            add_warning_log(f"MyCentroid : LAP failed; using SciPy assignment: {e}")
    rows, cols = linear_sum_assignment(cm)
    return np.column_stack((rows, cols)).astype(np.int32)


def _euclidean_dist_matrix(curr_pts: np.ndarray, prev_pts: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance between current points and previous points.

    Parameters
    ----------
    curr_pts : ``ndarray (Nc, 2)``
    prev_pts : ``ndarray (Np, 2)``

    Returns
    -------
    ndarray (Nc, Np) float32
    """
    # Broadcasting: (Nc,1,2) - (1,Np,2) -> (Nc,Np,2)
    diff = curr_pts[:, None, :] - prev_pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2, dtype=np.float32)
    return np.sqrt(d2, dtype=np.float32)


class MyCentroid(object):
    """Class representing a Centroid tracker."""

    def __init__(self, cfg):
        """Initialize according to the given ``cfg``.

        Parameters
        ----------
        cfg : TCFGCentroid
            A TCFGCentroid object which manages the configurations of tracker Centroid.
        """
        self.max_spread = cfg.max_spread
        self.previous_list: List[Person] = []
        self.current_list: List[Person] = []
        # Monotonic counter to prevent accidental CID reuse across frames
        self._next_cid = 0

    def __generateID__(self) -> int:
        """Generate a new unique tracking ID (CID)."""
        cid = self._next_cid
        self._next_cid += 1
        return cid

    def __findPID__(self, point) -> int:
        """Greedy NN fallback: find closest previous index to point within max_spread."""
        pindex = -1
        min_dist = 8192.0
        for i, p in enumerate(self.previous_list):
            if not hasattr(p, "repspoint") or p.repspoint is None:
                continue
            dist = hypot(p.repspoint[0] - point[0], p.repspoint[1] - point[1])
            if dist < min_dist:
                min_dist = dist
                pindex = i
        if min_dist > self.max_spread:
            pindex = -1
        return pindex

    def _lap_match(
        self,
        curr_points: np.ndarray,
        prev_points: np.ndarray,
        max_spread: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LAP-based matching with distance gating.

        Parameters
        ----------
        curr_points : (Nc,2) float32
        prev_points : (Np,2) float32
        max_spread : float

        Returns
        -------
        matches : (k,2) int32 -> [cur_idx, prev_idx]
        unmatched_curr : (u_c,) int32
        unmatched_prev : (u_p,) int32
        """
        Nc = curr_points.shape[0]
        Np = prev_points.shape[0]
        if Nc == 0 or Np == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.arange(Nc, dtype=np.int32),
                np.arange(Np, dtype=np.int32),
            )

        dist_mat = _euclidean_dist_matrix(curr_points, prev_points)

        # Gate distances above max_spread by assigning a large cost
        BIG = np.float32(1e6)
        cost_mat = dist_mat.copy()
        cost_mat[dist_mat > max_spread] = BIG

        pairs = _linear_assignment(cost_mat)
        if pairs is None or pairs.size == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.arange(Nc, dtype=np.int32),
                np.arange(Np, dtype=np.int32),
            )

        # Filter out pairs beyond max_spread
        keep = dist_mat[pairs[:, 0], pairs[:, 1]] <= max_spread
        matched = pairs[keep].astype(np.int32, copy=False)

        # Compute unmatched via masks
        cur_mask = np.ones(Nc, dtype=bool)
        prev_mask = np.ones(Np, dtype=bool)
        if matched.size:
            cur_mask[matched[:, 0]] = False
            prev_mask[matched[:, 1]] = False
        unmatched_curr = np.nonzero(cur_mask)[0].astype(np.int32, copy=False)
        unmatched_prev = np.nonzero(prev_mask)[0].astype(np.int32, copy=False)

        return matched, unmatched_curr, unmatched_prev

    def update(self, person_list, img=None):
        """Update the tracker and return the updated list of
        :class:`pyppbox.utils.persontools.Person`.

        Parameters
        ----------
        person_list : list[Person, ...]
            A list of :class:`pyppbox.utils.persontools.Person` object which stores
            the detected people in the given ``img``.
        img : object
            Defaults to ``None``.
            Being consistent with other trackers, will be ignored.

        Returns
        -------
        list[Person, ...]
            The updated list of :class:`pyppbox.utils.persontools.Person` object.

        Notes
        -----
        Call once per input frame and keep one tracker instance per stream. Person IDs,
        identity metadata, and ``misc`` can be updated in place; returned people are not
        independent snapshots. Use a result recorder or copy objects for historical data.
        Matching uses representative-point distances between adjacent frames.
        An empty frame clears the current detections, so a later detection receives a
        new CID. This tracker does not keep an age-based cache across empty frames.
        """
        self.previous_list = self.current_list
        self.current_list = []
        used_cids = set()

        if len(person_list) > 0:
            # Preserve original behavior: check only the first element type
            if isinstance(person_list[0], Person):
                self.current_list = person_list

                # Build index maps for points present
                prev_idx_map = []
                prev_pts = []
                for i, p in enumerate(self.previous_list):
                    rp = getattr(p, "repspoint", None)
                    if rp is not None:
                        prev_idx_map.append(i)
                        prev_pts.append([float(rp[0]), float(rp[1])])

                cur_idx_map = []
                cur_pts = []
                for i, p in enumerate(self.current_list):
                    rp = getattr(p, "repspoint", None)
                    if rp is not None:
                        cur_idx_map.append(i)
                        cur_pts.append([float(rp[0]), float(rp[1])])

                matched_pairs = []
                unmatched_cur_all = set(range(len(self.current_list)))

                # Prefer LAP-based matching when available
                if len(prev_pts) and len(cur_pts):
                    cur_arr = np.asarray(cur_pts, dtype=np.float32)
                    prev_arr = np.asarray(prev_pts, dtype=np.float32)
                    matched, unmatched_cur, _ = self._lap_match(cur_arr, prev_arr, float(self.max_spread))

                    # Translate back to original indices
                    for c_local, p_local in matched:
                        c_idx = cur_idx_map[int(c_local)]
                        p_idx = prev_idx_map[int(p_local)]
                        matched_pairs.append((c_idx, p_idx))
                        if c_idx in unmatched_cur_all:
                            unmatched_cur_all.remove(c_idx)
                    # Any current without repspoint are already in unmatched_cur_all
                    # Add currents with repspoint but unmatched by LAP:
                    for c_local in unmatched_cur:
                        c_idx = cur_idx_map[int(c_local)]
                        unmatched_cur_all.add(c_idx)

                # First pass: apply matches
                hang_indexes_in_clist = []
                for c_idx, p_idx in matched_pairs:
                    prev_p = self.previous_list[p_idx]
                    prev_cid = getattr(prev_p, "cid", None)
                    if prev_cid is not None and prev_cid not in used_cids:
                        self.current_list[c_idx].updateIDs(
                            prev_cid,
                            getattr(prev_p, "faceid", None),
                            getattr(prev_p, "deepid", None),
                            getattr(prev_p, "faceid_conf", None),
                            getattr(prev_p, "deepid_conf", None),
                        )
                        self.current_list[c_idx].misc = getattr(prev_p, "misc", None)
                        used_cids.add(prev_cid)
                    else:
                        hang_indexes_in_clist.append(c_idx)

                # Second pass: assign new IDs to unmatched or conflicted detections
                for c_idx in sorted(unmatched_cur_all.union(hang_indexes_in_clist)):
                    new_cid = self.__generateID__()
                    while new_cid in used_cids:
                        new_cid = self.__generateID__()
                    self.current_list[c_idx].cid = new_cid
                    used_cids.add(new_cid)
            else:
                msg = ("MyCentroid : update() -> The element of input 'person_list' list has unsupported type.")
                add_error_log(msg)
                raise ValueError(msg)

        return self.current_list
