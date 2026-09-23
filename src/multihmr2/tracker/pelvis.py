# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Per-track pelvis position predictors based on ridge regression."""

import torch
from .tools import Ridge, toquat, quat_batched_cdist_geodesic_deg
    
class RidgePredictor(object):

    def __init__(self, T=6, alpha=16.0):
        """Initialize the ridge regression predictor.

        Args:
            T: Maximum number of past observations to retain for fitting.
            alpha: Regularization strength for Ridge regression.
        """
        self.T = T
        self.alpha = alpha
        self.coords = None
        self.ridge = Ridge(alpha=alpha)
        
    def update_fitting(self, t, coords):
        """Add a new (time, coordinate) observation and refit the ridge regressor.

        The observation is appended to the internal buffer. If the buffer exceeds T
        entries, the oldest entry is discarded. The ridge model is then refit on all
        buffered observations.

        Args:
            t: Scalar 0-dimensional tensor representing the current time step.
            coords: 1D tensor of coordinates to record at time t.
        """
        assert t.numel()==1 and t.ndim==0, (t.size(), t.numel())
        assert coords.ndim==1, coords.size()
        newcoords = torch.cat( (t.view(1),coords), dim=0)[None,:]
        if self.coords is None:
            self.coords = newcoords.reshape(1, -1)
        else:
            self.coords = torch.cat( (self.coords,newcoords), dim=0)
        if self.coords.size(0)>self.T: self.coords = self.coords[-self.T:,:]
        self.ridge.fit(self.coords[:,0:1], self.coords[:,1:])
    
    def predict(self, t):
        """Predict coordinates at a given time step using the fitted ridge model.

        Args:
            t: Time step to predict at (scalar Python float or int).

        Returns:
            1D tensor of predicted coordinates.
        """
        ref = self.coords
        t_tensor = torch.tensor([[t]], device=ref.device, dtype=ref.dtype)
        return self.ridge.predict(t_tensor)[0, ...]

    def last_frame(self):
        """Return the time step of the most recently observed frame.

        Returns:
            Integer time step of the last observation stored in the buffer.
        """
        return int(self.coords[-1, 0])
    

def get_pelvis_fun(pelvis_mode, fun_name):
    """Return a similarity function for comparing pelvis positions.

    The returned function maps distance tensors to similarities in [0, 1] using
    exponential decay. The signature of the returned function depends on pelvis_mode:

    - 'xyz' + 'exp<a>': (dist,) -> exp(-dist/a)
    - 'ijn' + 'exp_<ap>_<an>': (pdist, ndist) -> exp(-pdist/ap)*exp(-ndist/an)
    - 'xyzQ' + 'exp_<ax>_<ao>': (dist, odist) -> exp(-dist/ax)*exp(-odist/ao)
    - 'ijnQ' + 'exp_<ap>_<an>_<ao>': (pdist, ndist, odist) -> product of three terms

    Args:
        pelvis_mode: One of 'xyz', 'ijn', 'xyzQ', 'ijnQ'.
        fun_name: String encoding the function type and parameters (e.g. 'exp1.0' or
            'exp_1.0_3.0_20.0').

    Returns:
        A callable with a signature matching the chosen pelvis_mode.

    Raises:
        ValueError: If the pelvis_mode and fun_name combination is not recognized.
    """
    if pelvis_mode=='xyz' and fun_name.startswith('exp'): # exp1.0
        alpha = float(fun_name[len('exp'):])
        dist_to_sim = lambda dist: torch.exp(-dist/alpha)
    elif pelvis_mode=='ijn' and fun_name.startswith('exp_'): # exp_1.0_3.0
        assert len(fun_name.split('_'))==3
        alpha_p = float(fun_name.split('_')[1])
        alpha_n = float(fun_name.split('_')[2])
        dist_to_sim = lambda pdist, ndist: torch.exp(-pdist/alpha_p) * torch.exp(-ndist/alpha_n)
    elif pelvis_mode=='xyzQ' and fun_name.startswith('exp_'):  # exp_1.0_3.0
        assert len(fun_name.split('_'))==3
        alpha_x = float(fun_name.split('_')[1])
        alpha_o = float(fun_name.split('_')[2])
        dist_to_sim = lambda dist, odist: torch.exp(-dist/alpha_x) * torch.exp(-odist/alpha_o)
    elif pelvis_mode=='ijnQ' and fun_name.startswith('exp_'):  # exp_1.0_3.0_20.0
        assert len(fun_name.split('_'))==4
        alpha_p = float(fun_name.split('_')[1])
        alpha_n = float(fun_name.split('_')[2])
        alpha_o = float(fun_name.split('_')[3])
        dist_to_sim = lambda pdist, ndist, odist: torch.exp(-pdist/alpha_p) * torch.exp(-ndist/alpha_n) * torch.exp(-odist/alpha_o)
    else:
        raise ValueError(f'Unknown pelvis distance-to-similarity function for {pelvis_mode=} and pelvis_fun={fun_name}')
    return dist_to_sim
    
class PelvisPredictorBank(object):
    
    def __init__(self, pelvis_mode, pelvis_fun, pelvis_pred, max_tskip):
        """Initialize the pelvis predictor bank.

        Args:
            pelvis_mode: Coordinate representation for the pelvis. One of 'xyz',
                'ijn', 'xyzQ', 'ijnQ'.
            pelvis_fun: Name of the distance-to-similarity function (e.g.
                'exp_1.0_3.0').
            pelvis_pred: Predictor type string. Only 'ridgeT<T>A<alpha>' is
                supported, which creates a RidgePredictor(T=T, alpha=alpha) per
                track.
            max_tskip: Maximum number of frames a track can be absent before
                it is removed.
        """
        assert pelvis_mode in ['xyz','ijn','xyzQ','ijnQ'], f'[{self.__class__.__name__}] Unknown pelvis mode {pelvis_mode}'
        self.pelvis_mode = pelvis_mode
        self.max_tskip = max_tskip
        self.pelvis_func = get_pelvis_fun(pelvis_mode, pelvis_fun)
        if pelvis_pred.startswith('ridge'): # ridgeT6A16.0
            ridgeT = int(pelvis_pred[len('ridgeT'):].split('A')[0])
            ridgeAlpha = float(pelvis_pred.split('A')[1])
            self.build_new_predictor = lambda: RidgePredictor(T=ridgeT,alpha=ridgeAlpha)
        else:
            raise ValueError(f'Unknown {pelvis_pred=}')
    
    
    def reset(self):
        """Clear all tracked predictors."""
        self.predictors = {}

    def update_without_detection(self, t):
        """Advance time without new detections, pruning stale tracks.

        Removes all predictors whose last observed frame is more than max_tskip
        frames before t+1.

        Args:
            t: Current time step.
        """
        self.predictors = {hid: pred for hid, pred in self.predictors.items()
                           if pred.last_frame() > t + 1 - self.max_tskip}

    def init(self, t, human_ids, pelvis_xyz=None, pelvis_ijn=None, pelvis_rotmat=None):
        """Initialize predictors for a set of newly observed persons.

        Delegates to update, creating new predictors for each human ID.

        Args:
            t: Current time step.
            human_ids: 1D tensor of human identity indices.
            pelvis_xyz: Pelvis 3D positions (optional).
            pelvis_ijn: Pelvis pixel+nearness positions (optional).
            pelvis_rotmat: Pelvis rotation matrices (optional).
        """
        self.update(t, human_ids, pelvis_xyz=pelvis_xyz, pelvis_ijn=pelvis_ijn, pelvis_rotmat=pelvis_rotmat)

    def similarity(self, t, hid, pelvis_xyz=None, pelvis_ijn=None, pelvis_rotmat=None):
        """Compute similarity between current detections and all tracked persons.

        For each tracked person in hid, the predictor predicts coordinates at time t.
        Similarity is computed as an exponential function of the distance(s) between
        predicted and observed pelvis coordinates.

        Args:
            t: Current time step (scalar).
            hid: 1D tensor of tracked human IDs to compare against.
            pelvis_xyz: Observed 3D pelvis positions of shape (N, 3) (used in
                'xyz'/'xyzQ' modes).
            pelvis_ijn: Observed pixel+nearness positions of shape (N, 3) (used in
                'ijn'/'ijnQ' modes).
            pelvis_rotmat: Observed pelvis rotation matrices of shape (N, 3, 3)
                (used in 'xyzQ'/'ijnQ' modes).

        Returns:
            Similarity tensor of shape (N, len(hid)).
        """
        pred_coor = torch.stack( [self.predictors[int(i)].predict(t) for i in hid], dim=0) # hid.size(0) x 3 (or 7 with quat)
        if self.pelvis_mode.endswith('Q'): pelvis_quat = toquat(pelvis_rotmat).float()
        if self.pelvis_mode=='xyz':
            assert pelvis_xyz is not None
            dist = torch.cdist( pelvis_xyz[None,...], pred_coor[None,...] )[0,:,:]
            sim = self.pelvis_func(dist)
        elif self.pelvis_mode=='ijn':
            assert pelvis_ijn is not None
            pdist = torch.cdist( pelvis_ijn[None,:,0:2], pred_coor[None,:,0:2] )[0,:,:]
            ndist = torch.cdist( pelvis_ijn[None,:,2:3], pred_coor[None,:,2:3] )[0,:,:]
            sim = self.pelvis_func(pdist, ndist)
        elif self.pelvis_mode=='xyzQ':
            assert pelvis_xyz is not None and pelvis_rotmat is not None
            dist = torch.cdist( pelvis_xyz[None,...], pred_coor[None,...,0:3] )[0,:,:]
            odist = torch.cdist( pelvis_quat[None,...], pred_coor[None,...,3:7] )[0,:,:]
            sim = self.pelvis_func(dist, odist)
        elif self.pelvis_mode=='ijnQ':
            assert pelvis_ijn is not None and pelvis_rotmat is not None
            pdist = torch.cdist( pelvis_ijn[None,:,0:2], pred_coor[None,:,0:2] )[0,:,:]
            ndist = torch.cdist( pelvis_ijn[None,:,2:3], pred_coor[None,:,2:3] )[0,:,:]
            odist = quat_batched_cdist_geodesic_deg( pelvis_quat[None,:,:], pred_coor[None,:,3:7] )[0,:,:]
            sim = self.pelvis_func(pdist, ndist, odist)            
        return sim

    def update(self, t, human_ids, pelvis_xyz=None, pelvis_ijn=None, pelvis_rotmat=None):
        """Update the predictor for each tracked person with a new observation.

        For each human ID, creates a new predictor if not already tracked, then
        appends the current observation.

        Args:
            t: Current time step (int or tensor matching human_ids shape).
            human_ids: 1D tensor of human identity indices.
            pelvis_xyz: Observed 3D pelvis positions (used in 'xyz'/'xyzQ' modes).
            pelvis_ijn: Observed pixel+nearness positions (used in 'ijn'/'ijnQ'
                modes).
            pelvis_rotmat: Observed pelvis rotation matrices (used in 'xyzQ'/
                'ijnQ' modes).
        """
        if isinstance(t,int): t = t*torch.ones_like(human_ids)
        pelvis_coord = pelvis_xyz if self.pelvis_mode.startswith('xyz') else pelvis_ijn
        if self.pelvis_mode.endswith('Q'):
            pelvis_coord = torch.cat( (pelvis_coord,toquat(pelvis_rotmat).float()), dim=1)
        for i in range(len(human_ids)):
            hid = int(human_ids[i])
            if hid not in self.predictors: self.predictors[hid] = self.build_new_predictor()
            self.predictors[hid].update_fitting(t[i], pelvis_coord[i])