"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import glob
import time
import argparse

import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

from filterpy.kalman import KalmanFilter

import lap

np.random.seed(0)


def linear_assignment(cost_matrix):
  """
  LAPX wrapper with best practices:
  - Ensures float32 and C-contiguous
  - Uses extend_cost only when the matrix is rectangular
  - Returns (k,2) int32 [row, col] pairs
  """
  cm = np.ascontiguousarray(cost_matrix, dtype=np.float32)
  extend = (cm.shape[0] != cm.shape[1])
  pairs = lap.lapjvxa(cm, extend_cost=extend, return_cost=False)
  # if pairs is None:
  #   return np.empty((0, 2), dtype=np.int32)
  # pairs = np.asarray(pairs, dtype=np.int32)
  # if pairs.ndim != 2 or pairs.shape[1] != 2:
  #   pairs = pairs.reshape((-1, 2)).astype(np.int32, copy=False)
  return pairs


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IoU between two bboxes in the form [x1,y1,x2,y2]
  Returns ndarray (len(bb_test), len(bb_gt)) float32
  """
  bb_gt = np.expand_dims(bb_gt, 0).astype(np.float32, copy=False)
  bb_test = np.expand_dims(bb_test, 1).astype(np.float32, copy=False)

  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

  w = np.maximum(0.0, xx2 - xx1)
  h = np.maximum(0.0, yy2 - yy1)
  inter = w * h

  area_a = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
  area_b = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
  union = area_a + area_b - inter

  with np.errstate(divide="ignore", invalid="ignore"):
    o = np.where(union > 0.0, inter / union, 0.0).astype(np.float32)
  # Sanitize any potential NaN/Inf
  o[~np.isfinite(o)] = 0.0
  return o


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w / 2.0
  y = bbox[1] + h / 2.0
  s = w * h    # scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] (or [x1,y1,x2,y2,score] if score is not None)
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if score is None:
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
  else:
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0

  def __init__(self, bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    # define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]])
    self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0]])

    self.kf.R[2:, 2:] *= 10.0
    self.kf.P[4:, 4:] *= 1000.0  # high uncertainty to unobservable initial velocities
    self.kf.P *= 10.0
    self.kf.Q[-1, -1] *= 0.01
    self.kf.Q[4:, 4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if (self.kf.x[6] + self.kf.x[2]) <= 0:
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if self.time_since_update > 0:
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
  """
  Assigns detections to tracked objects (both as bounding boxes).
  Returns:
    matches: ndarray (k,2) int32 -> [det_index, trk_index]
    unmatched_detections: ndarray (u_d,) int32
    unmatched_trackers: ndarray (u_t,) int32
  """
  detections = np.asarray(detections)
  trackers = np.asarray(trackers)

  Nd = len(detections)
  Nt = len(trackers)
  if Nt == 0:
    return np.empty((0, 2), dtype=np.int32), np.arange(Nd, dtype=np.int32), np.empty((0,), dtype=np.int32)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    feasible = (iou_matrix >= iou_threshold).astype(np.int32)

    # Fast path: unique-feasible (≤ 1 per row and col)
    if feasible.any() and feasible.sum(1).max() == 1 and feasible.sum(0).max() == 1:
      r, c = np.nonzero(feasible)
      matched_indices = np.stack([r.astype(np.int32), c.astype(np.int32)], axis=1) if r.size else np.empty((0, 2), np.int32)
    else:
      # print(f"iou_matrix:\n{iou_matrix.shape}")
      # t0 = time.time()
      pairs = linear_assignment(1.0 - iou_matrix)
      matched_indices = pairs.astype(np.int32, copy=False) if pairs is not None else np.empty((0, 2), dtype=np.int32)
      # t1 = time.time()
      # print(f" Matching time: {(t1 - t0) * 1000.0:.6f} ms")
  else:
    matched_indices = np.empty((0, 2), dtype=np.int32)

  # Compute unmatched using boolean masks (fast and correct)
  det_mask = np.ones(Nd, dtype=bool)
  trk_mask = np.ones(Nt, dtype=bool)
  if matched_indices.size:
    det_mask[matched_indices[:, 0]] = False
    trk_mask[matched_indices[:, 1]] = False
  unmatched_detections = np.nonzero(det_mask)[0].astype(np.int32, copy=False)
  unmatched_trackers = np.nonzero(trk_mask)[0].astype(np.int32, copy=False)

  # Filter out matched with low IoU (tracking semantics)
  matches = []
  for m in matched_indices:
    if iou_matrix[m[0], m[1]] < iou_threshold:
      unmatched_detections = np.append(unmatched_detections, m[0]).astype(np.int32, copy=False)
      unmatched_trackers = np.append(unmatched_trackers, m[1]).astype(np.int32, copy=False)
    else:
      matches.append(m.reshape(1, 2))
  if len(matches) == 0:
    matches = np.empty((0, 2), dtype=np.int32)
  else:
    matches = np.concatenate(matches, axis=0)

  # Deduplicate in case any indices reappeared during post-filter
  if unmatched_detections.size:
    unmatched_detections = np.unique(unmatched_detections)
  if unmatched_trackers.size:
    unmatched_trackers = np.unique(unmatched_trackers)

  return matches, unmatched_detections, unmatched_trackers


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score], ...]
    Requires: call once per frame even with empty detections (use np.empty((0, 5))).
    Returns an array where the last column is the object ID.
    """
    self.frame_count += 1

    # Predict current tracker states
    trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # Update matched trackers with detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # Create new trackers for unmatched detections
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i, :])
      self.trackers.append(trk)

    # Prepare return and cull old trackers
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state()[0]
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
      i -= 1
      if trk.time_since_update > self.max_age:
        self.trackers.pop(i)

    if len(ret) > 0:
      return np.concatenate(ret)
    return np.empty((0, 5))

  def update_pyppbox(self, current_people):
    """
    Similar to update(). Made for pyppbox's Person list.
    """
    self.frame_count += 1

    # Build detections efficiently
    if len(current_people) == 0:
      dets = np.empty((0, 5), dtype=np.float32)
    else:
      rows = [p.getDetRS() for p in current_people]
      dets = np.vstack(rows).astype(np.float32, copy=False)

    # Predict current tracker states
    trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
    to_del = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    people_trackers = []
    updated_people = []

    # Update matched trackers with detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])
      p = current_people[m[0]]
      p.cid = self.trackers[m[1]].id
      people_trackers.append([p, m[1]])

    # Create and initialise new trackers for unmatched detections
    for u in unmatched_dets:
      trk = KalmanBoxTracker(dets[u, :])
      self.trackers.append(trk)
      if u < len(current_people):
        p = current_people[u]
        p.cid = trk.id
        people_trackers.append([p, len(self.trackers) - 1])

    # Emit updated people for trackers updated this frame
    for pp_trk in people_trackers:
      trk = self.trackers[pp_trk[1]]
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        updated_people.append(pp_trk[0])

    # IMPORTANT: cull dead tracklets globally (prevents Nt blow-up)
    for i in reversed(range(len(self.trackers))):
      if self.trackers[i].time_since_update > self.max_age:
        self.trackers.pop(i)

    return updated_people


def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='SORT demo')
  parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]', action='store_true')
  parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
  parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
  parser.add_argument("--max_age", help="Maximum number of frames to keep alive a track without associated detections.", type=int, default=1)
  parser.add_argument("--min_hits", help="Minimum number of associated detections before track is initialised.", type=int, default=3)
  parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3)  # used only for display
  if display:
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n'
            '    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n'
            '    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

    with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
      print("Processing %s." % (seq))
      for frame in range(int(seq_dets[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # convert [x1,y1,w,h] -> [x1,y1,x2,y2]
        total_frames += 1

        if display:
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))
          im = io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
          if display:
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))

        if display:
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if display:
      print("Note: to get real runtime results run without the option: --display")
