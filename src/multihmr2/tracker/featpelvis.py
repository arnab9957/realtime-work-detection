# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Multi-person tracker combining feature similarity and pelvis prediction."""

import torch

from . import tools as trackertools
from .featbank import FeatureBank
from .pelvis import PelvisPredictorBank

class FeatPelvisTracker(object):

    def __init__(self,
        # assignment hyperparameters
        max_tskip=50,                 # maximum number of missed frames before deleting an old track
        min_sim=0.69,                 # similarity threshold below which a new track is started instead of being assigned
        # feature similarity hyperparameters
        feat_aggsim='Knn35t15.0',     # how to aggregate similarity of features with those of the last frames, only knn<K>t<T> implemented where K is the number of K-NN and a temperature of T
        feat_l2norm=True,            # L2-normalize features before computing distances
        feat_pdist=2,                 # pdist to compute distances between feature
        # pelvis hyperparameters
        pelvis_mode='ijnQ',           # can be 'ijn' for pixel&nearness or 'xyz' for 3d coordinate, ijnQ/xyzQ for adding orientation
        pelvis_fun='exp_1.0_3.0_20.0',# how to mesure distances between pelvis positions (typically 'exp_<a>_<b>_<c>' for mode 'ijnQ', exp_<a>_<b> for mode ijn/xyzQ or 'exp<a>' for mode xyz)
        pelvis_pred='ridgeT8A20.0',   # how to predict the position of the pelvis in features frames, either 'last' or 'ridgeT<T>A<alpha>'
        # combination feature+pelvis hyperparameters
        combine='wsum_1.0_8.0',      # how to combine the feature and pelvis similarity, e.g. ='wsum_1.0_10.0' for weighted sum of 1.0*feat+10.0*pelvis
    ):
        """Initialize the FeatPelvisTracker with feature and pelvis banks.

        Args:
            max_tskip: Maximum number of missed frames before a track is deleted.
            min_sim: Similarity threshold below which a detection starts a new
                track instead of being assigned to an existing one.
            feat_aggsim: How to aggregate feature similarity across stored frames;
                format is 'Knn<K>t<T>' where K is the number of nearest neighbors
                and T is the softmax temperature.
            feat_l2norm: If True, L2-normalizes features before computing distances.
            feat_pdist: p-norm used for feature distance computation.
            pelvis_mode: Coordinate representation for pelvis positions ('xyz',
                'ijn', 'xyzQ', or 'ijnQ').
            pelvis_fun: Distance-to-similarity function name for pelvis positions.
            pelvis_pred: Pelvis predictor type (e.g. 'ridgeT6A16.0').
            combine: How to combine feature and pelvis similarities; format is
                'wsum_<wf>_<wp>' for a weighted sum with weights normalized to
                sum to 1.
        """
        # assignment hyperparameters
        self.max_tskip = max_tskip
        self.min_sim = min_sim
        # for features similarity
        self.feature_bank = FeatureBank(feat_aggsim, feat_l2norm, feat_pdist, max_tskip)
        # for pelvis similarity
        self.pelvis_bank = PelvisPredictorBank(pelvis_mode, pelvis_fun, pelvis_pred, max_tskip)
        # combination feature+pelvis hyperparameters
        self.combine_fun = trackertools.get_combine_function(combine)
        # reset
        self.reset()

    def reset(self):
        """Reset all internal state: next_human_id counter, feature bank, and pelvis bank."""
        self.next_human_id = 0
        self.feature_bank.reset()
        self.pelvis_bank.reset()

    def is_empty(self):
        """Return True if the feature bank has no stored tracks."""
        return self.feature_bank.is_empty()

    def init_tracker(self, t, features, pelvis_xyz, pelvis_ijn, pelvis_ori):
        """Initialize the tracker with the first set of detections.

        Assigns consecutive IDs starting from next_human_id, populates the feature
        and pelvis banks, and advances next_human_id.

        Args:
            t: Current time step (scalar).
            features: Feature vectors of shape (N, D) for the N detected persons.
            pelvis_xyz: Pelvis 3D positions of shape (N, 3).
            pelvis_ijn: Pelvis pixel+nearness positions of shape (N, 3).
            pelvis_ori: Pelvis rotation matrices of shape (N, 3, 3).

        Returns:
            1D long tensor of assigned human IDs of shape (N,).
        """
        human_ids = self.next_human_id + torch.arange(features.size(0), device=features.device)
        T = t*torch.ones_like(human_ids)
        self.feature_bank.init(T, human_ids, features)
        self.pelvis_bank.init(T, human_ids, pelvis_xyz, pelvis_ijn, pelvis_ori)
        self.next_human_id += features.size(0)
        return human_ids

    def _compute_assignment(self, sim, hid):
        """Assign detections to tracked identities using Hungarian matching.

        Detections whose best match has similarity below min_sim receive new IDs.
        New IDs are allocated from next_human_id sequentially.

        Args:
            sim: Similarity matrix of shape (num_det, num_tracked).
            hid: 1D tensor of tracked human IDs corresponding to columns of sim.

        Returns:
            pred_ids: 1D long tensor of assigned IDs for each detection.
            is_new_pred_ids: Boolean tensor of shape (num_det,), True for detections
                that received a new ID.
        """
        sim2 = sim.clone()
        pred_ids = -torch.ones( (sim.size(0),), dtype=torch.long, device=sim.device)
        row_indices, col_indices = trackertools.hungarian_matching(1-sim2, max_threshold=1-self.min_sim, max_val=1-self.min_sim+1e-5)
        for i in range(len(row_indices)):
            pred_ids[row_indices[i]] = hid[col_indices[i]]
        # new ids
        is_new_pred_ids = pred_ids<0
        if torch.any(is_new_pred_ids):
            has_new_ids = torch.where(pred_ids<0)[0]
            pred_ids[has_new_ids] = self.next_human_id+torch.arange(len(has_new_ids), device=sim.device)
            self.next_human_id += len(has_new_ids)
        return pred_ids, is_new_pred_ids
        
    def track_next_frame(self, t, features=None, pelvis_xyz=None, pelvis_ijn=None, pelvis_ori=None):
        """Process one frame: update the feature/pelvis banks and return assigned IDs.

        Handles three cases:
        1. No detections: updates banks for time advancement without new data.
        2. Empty banks (first detections ever): initializes the tracker.
        3. Normal case: computes combined feature+pelvis similarity, assigns IDs via
           Hungarian matching, and updates banks.

        Args:
            t: Current time step (scalar).
            features: Feature vectors of shape (N, D) for detected persons.
            pelvis_xyz: Pelvis 3D positions of shape (N, 3).
            pelvis_ijn: Pelvis pixel+nearness positions of shape (N, 3).
            pelvis_ori: Pelvis rotation matrices of shape (N, 3, 3).

        Returns:
            1D long tensor of assigned human IDs of shape (N,).
        """
        # Special case: no detection at this frame
        if features.size(0)==0:
            self.feature_bank.update_without_detection(t)
            self.pelvis_bank.update_without_detection(t)
            return torch.empty((0,), dtype=torch.long, device=features.device)
        # Special case where is no "past"
        if self.is_empty():
            return self.init_tracker(t, features, pelvis_xyz, pelvis_ijn, pelvis_ori)
        # get similarity from the features and pelvis, and combine them 
        feat_sim, bank_human_ids = self.feature_bank.similarity(features)
        pelvis_sim = self.pelvis_bank.similarity(t, bank_human_ids, pelvis_xyz, pelvis_ijn, pelvis_ori)
        sim = self.combine_fun(feat_sim, pelvis_sim)
        # compute assignment 
        human_ids, is_new_human_ids = self._compute_assignment(sim, bank_human_ids)
        # update banks
        self.feature_bank.update(t, human_ids, features)
        self.pelvis_bank.update(t, human_ids, pelvis_xyz, pelvis_ijn, pelvis_ori)
        return human_ids