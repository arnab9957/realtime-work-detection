# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Bank of recent per-track tracking features for similarity matching."""

import torch
import torch.nn.functional as F

class FeatureBank(object):

    def __init__(self, aggsim='Knn30t20.0', l2norm=False, pdist=1, max_tskip=50):
        """Initialize the feature bank.

        Args:
            aggsim: Aggregation method for per-track similarity. Only 'Knn<K>t<T>'
                is supported, using the K nearest neighbors with softmax temperature
                T.
            l2norm: If True, L2-normalizes features before storing or comparing.
            pdist: p-norm exponent for torch.cdist distance computation.
            max_tskip: Maximum number of frames after which stored features are
                considered stale and discarded.
        """
        self.aggsim = aggsim # how to aggregate the distance to different features of the same track
        if aggsim.startswith('Knn'): # eg Knn10t0.1
            self.feat_aggsim_K = int(aggsim[3:].split('t')[0])
            self.feat_aggsim_T = float(aggsim.split('t')[1])
        else:
            raise ValueError(f'[self.__class__.__name__] Unknown feature similarity aggregator {aggsim}')
        self.l2norm = l2norm            
        self.pdist = pdist
        self.max_tskip = max_tskip
        
    def reset(self):
        """Clear all stored features, IDs, and frame indices."""
        self.feat = None
        self.hid = None
        self.frame = None

    def is_empty(self):
        """Return True if the bank contains no stored features."""
        return self.feat is None

    def _normalize_if_needed(self, features):
        """L2-normalize features along dimension 1 if self.l2norm is True.

        Args:
            features: Input tensor of shape (N, D).

        Returns:
            Normalized or unchanged features tensor.
        """
        if self.l2norm:
            return torch.nn.functional.normalize(features, dim=1)
        return features

    def init(self, t, human_ids, features):
        """Initialize the bank with the first set of features.

        Args:
            t: 1D tensor of frame indices, one per feature vector.
            human_ids: 1D tensor of human identity indices.
            features: Feature tensor of shape (N, D).
        """
        features = self._normalize_if_needed(features)
        self.feat = features.clone()
        self.hid = human_ids.clone()
        self.frame = t.clone()

    def similarity(self, features):
        """Compute similarity between query features and all stored features.

        Distances are computed with torch.cdist (p=self.pdist). For each query, the
        K nearest stored features are found and their distances are converted to
        similarities via softmax with temperature self.feat_aggsim_T. Similarities
        are then aggregated per unique human ID.

        Args:
            features: Query feature tensor of shape (N, D).

        Returns:
            agg_sim: Similarity tensor of shape (N, num_unique_ids).
            unique_hid: 1D tensor of unique human IDs corresponding to columns of
                agg_sim.
        """
        features = self._normalize_if_needed(features)
        # compute distances between these features and the bank
        cdist = torch.cdist( features[None,:,:], self.feat[None,:,:], p=self.pdist)[0,:,:] # nfeatures x nfeat 
        # get the list of unique human ids in the bank 
        unique_hid = torch.unique(self.hid)
        # build similarity
        if self.aggsim.startswith('Knn'):
            topk_dist, indices = cdist.topk( min(self.feat_aggsim_K,cdist.size(1)), largest=False, sorted=True)
            topk_hid = torch.gather(self.hid[None,:].repeat(cdist.size(0),1), 1, indices)
            topk_sims_transform = F.softmax(-topk_dist / self.feat_aggsim_T, 1)
            scores = torch.mul( F.one_hot(topk_hid, num_classes=unique_hid.max()+1), topk_sims_transform[:,:,None]).sum(dim=1)
            agg_sim = scores[:,unique_hid]
            assert torch.all(agg_sim >= -1e-5) and torch.all(agg_sim <= 1 + 1e-5), (
                f"agg_sim out of expected [0, 1] range: "
                f"min={agg_sim.min().item()}, max={agg_sim.max().item()}"
            )
        else:
            raise ValueError(f'[self.__class__.__name__] Unknown feature similarity aggregator {self.aggsim}')
        return agg_sim, unique_hid       
    
    def update(self, t, human_ids, features):
        """Add new features to the bank and discard entries older than max_tskip frames.

        Args:
            t: Scalar time step for the new features.
            human_ids: 1D tensor of human identity indices for the new features.
            features: Feature tensor of shape (N, D).
        """
        features = self._normalize_if_needed(features)
        # delete features that are too old and add new ones
        tokeep = self.frame>t+1-self.max_tskip
        self.feat = torch.cat( (self.feat[tokeep,:],features), dim=0)
        self.hid = torch.cat( (self.hid[tokeep],human_ids), dim=0)
        self.frame = torch.cat( (self.frame[tokeep],t*torch.ones_like(human_ids)),dim=0)
            
    def update_without_detection(self, t):
        """Advance time without new features, discarding entries older than max_tskip.

        If no entries remain after pruning, the bank is fully reset.

        Args:
            t: Current time step (scalar).
        """
        if self.feat is not None: # nothing to do if no active tracks
            # delete features that are too old
            tokeep = self.frame>t+1-self.max_tskip
            if tokeep.sum()==0:
                self.reset()
            else:
                self.feat = self.feat[tokeep,:]
                self.hid = self.hid[tokeep]
                self.frame = self.frame[tokeep]