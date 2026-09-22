# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""DETR-style decoder producing per-person body pose, shape, and 3D position."""

import roma
import anny
import torch
import numpy as np
import json
from pathlib import Path

from torch import nn, Tensor
from dataclasses import dataclass, fields

from .encoder import EncoderOutput
from .hph import HPH
from .pos_embed import get_2d_sincos_pos_embed
from ...utils import (
    unpatch,
    inverse_perspective_projection,
    perspective_projection,
    rotation_to_homogeneous,
)


@dataclass
class DecoderConfig:
    xat_dim: int = 1024  # XAT dim
    xat_depth: int = 8  # number of cross attention block (SA, CA, MLP) in the HPH head.
    xat_heads: int = 16  # Number of attention heads
    xat_dim_head: int = 32
    xat_mlp_dim: int = 4 * 512
    xat_dropout: float = 0.0
    num_queries: int = 100
    person_center: str = "vtx"
    num_betas: int = 11
    trackfeat: str = "sam2.1_hiera_large__pooler_avg8"
    trackfeat_arch: str = "MLp"
    all_phenotypes: bool = False


@dataclass
class PersonOutput:
    num_person: int
    conf: Tensor  # (N, 1)
    shape: Tensor  # (N, num_betas)
    v3d: Tensor  # (N, V, 3)
    j3d: Tensor  # (N, J, 3)
    j2d: Tensor  # (N, J, 2)
    transl_pelvis: Tensor  # (N, 3)
    bone_poses: Tensor
    rest_bone_poses: Tensor
    track_id: Tensor | None = None
    trackfeat: Tensor | None = None

    def _apply(self, fn):
        """Apply a function to every Tensor field and rebuild a PersonOutput.

        Non-tensor fields are carried over unchanged, except num_person which is
        recomputed as len(conf) after the function is applied.

        Args:
            fn: Callable applied element-wise to each Tensor field.

        Returns:
            A new PersonOutput with fn applied to all tensor fields.
        """
        data = {}
        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, Tensor):
                data[f.name] = fn(value)
            else:
                data[f.name] = value

        data["num_person"] = len(data["conf"])
        return PersonOutput(**data)

    def __getitem__(self, idx):
        """Index all tensor fields along the first dimension.

        Args:
            idx: Index or slice applied to each tensor field.

        Returns:
            A new PersonOutput with the indexed tensors.
        """
        return self._apply(lambda x: x[idx])

    def to(self, device):
        """Move all tensor fields to the specified device.

        Args:
            device: Target device string or torch.device.

        Returns:
            A new PersonOutput with all tensors on device.
        """
        return self._apply(lambda x: x.to(device))


@dataclass
class DecoderOutput:
    K: Tensor  # (3, 3)
    persons: PersonOutput

    def to(self, device):
        """Move K and all person tensors to the specified device.

        Args:
            device: Target device string or torch.device.

        Returns:
            A new DecoderOutput with tensors on device.
        """
        return DecoderOutput(
            K=self.K.to(device),
            persons=self.persons.to(device),
        )

    def __len__(self):
        """Return the number of detected persons."""
        return self.persons.num_person


def nms_3d_pelvis(persons: PersonOutput, dist_thresh: float) -> PersonOutput:
    """Apply non-maximum suppression in 3D pelvis space.

    Persons are processed in descending confidence order. A person is suppressed if
    its pelvis position is within dist_thresh (Euclidean distance) of any
    already-kept person.

    Args:
        persons: PersonOutput with conf and transl_pelvis fields populated.
        dist_thresh: Distance threshold below which two detections are considered
            duplicates.

    Returns:
        A filtered PersonOutput containing only the kept persons.
    """
    if persons.num_person < 2:
        return persons

    keep = []
    pelvis = persons.transl_pelvis
    confs = persons.conf.squeeze(-1)
    used = torch.zeros_like(confs).bool()  # (N,)

    _, order = confs.sort(descending=True)
    for i in order:
        if used[i].item():
            continue
        keep.append(i.item())
        used[i] = True
        dist = torch.norm(pelvis - pelvis[i], dim=1)  # (N,)
        used |= dist < dist_thresh
    return persons[keep]


class Decoder(nn.Module):
    """A ViT backbone followed by a "HPH" head (stack of cross attention layers with a fixed number of queries.)"""

    def __init__(
        self,
        config: DecoderConfig,
        grid_size: int,
        patch_size: int,
        embed_dim: int,
    ):
        """Build the full-body DETR decoder head.

        Creates the HPH cross-attention transformer, positional embeddings, learnable
        queries, two Anny body models (full-resolution and low-resolution), and all
        regression heads (location, confidence, pose, shape, distance, tracking
        features).

        Args:
            config: Decoder configuration.
            grid_size: Spatial grid size of the encoder feature map (height and
                width in patches).
            patch_size: Stride of each patch in pixels.
            embed_dim: Channel dimension of the encoder feature map.
        """
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_betas = config.num_betas
        self.trackfeat = config.trackfeat
        self.num_queries = config.num_queries
        self.person_center = config.person_center

        # HumanPerceptionHead
        self.decoder = HPH(
            dim=config.xat_dim,
            depth=config.xat_depth,
            heads=config.xat_heads,
            dim_head=config.xat_dim_head,
            mlp_dim=config.xat_mlp_dim,
            dropout=config.xat_dropout,
        )

        # Positional/Token embedding
        dec_pos_emb = get_2d_sincos_pos_embed(
            embed_dim=config.xat_dim, grid_size=self.grid_size
        )
        dec_pos_emb = torch.from_numpy(dec_pos_emb).float()
        dec_pos_emb = unpatch(
            dec_pos_emb.unsqueeze(0),
            patch_size=1,
            img_size=int(np.sqrt(dec_pos_emb.shape[0])),
        ).permute(0, 2, 3, 1)
        self.register_buffer("dec_pos_emb", dec_pos_emb)
        self.dec_to_token = nn.Linear(embed_dim, config.xat_dim)

        # Queries
        self.queries = nn.Embedding(config.num_queries, self.decoder.dim)

        # Top-heads
        # 2D location
        self.mlp_loc = nn.Sequential(
            *[
                nn.Linear(self.decoder.dim, self.decoder.dim),
                nn.ReLU(),
                nn.Linear(self.decoder.dim, 2),
            ]
        )

        # Confidence
        self.mlp_conf = nn.Sequential(
            *[
                nn.Linear(self.decoder.dim, self.decoder.dim),
                nn.ReLU(),
                nn.Linear(self.decoder.dim, 1),
            ]
        )

        # Parametric 3D model
        self.body_model = anny.create_fullbody_model(
            local_changes=True,
            pose_parameterization="root_relative",
            remove_unattached_vertices=False,
            all_phenotypes=config.all_phenotypes,
        ).to(dtype=torch.float32)
        self.lowres_body_model = anny.create_fullbody_model(
            local_changes=True,
            pose_parameterization="root_relative",
            remove_unattached_vertices=False,
            all_phenotypes=config.all_phenotypes,
            topology="notoes_collapse5pc",
        ).to(dtype=torch.float32)
        assert self.person_center == "vtx", "Only person_center='vtx' is supported"
        self.person_center_idx = 5054
        self.person_center_idx_lowres = 9

        self.n_joints = 163
        self.body_model.set_skinning_method("lbs")
        self._keywords_to_discard = [
            "jaw",
            "oris",
            "levator",
            "orbicularis",
            "temporalis",
            "oculi",
            "risorius",
            "special",
            "toe",
            "eye",
            "tongue",
            "neck",
            "head",
            "foot",
            "breast",
        ]
        self._indices_to_discard = [
            i
            for i, b in enumerate(self.body_model.bone_labels)
            if any(key in b for key in self._keywords_to_discard)
        ]

        self.eye = nn.Parameter(torch.eye(3).unsqueeze(0), requires_grad=False)

        # Human properties
        self.mlp_pose = nn.Sequential(
            *[
                nn.Linear(self.decoder.dim + self.n_joints * 6, self.decoder.dim),
                nn.ReLU(),
                nn.Linear(self.decoder.dim, self.n_joints * 6),
            ]
        )
        self.mlp_shape = nn.Sequential(
            *[
                nn.Linear(self.decoder.dim, self.decoder.dim),
                nn.ReLU(),
                nn.Linear(self.decoder.dim, self.num_betas),
            ]
        )
        self.mlp_dist = nn.Sequential(
            *[
                nn.Linear(self.decoder.dim, self.decoder.dim),
                nn.ReLU(),
                nn.Linear(self.decoder.dim, 1),
            ]
        )

        # SMPL init
        # for root it should be [np.pi/2.,0,0.]
        init_root_pose = (
            roma.rotvec_to_rotmat(torch.tensor([[np.pi / 2.0, 0, 0.0]]))[:, :, :2]
            .flatten(1)
            .reshape(1, -1)
        )
        init_body_pose = (
            torch.eye(3)
            .reshape(1, 3, 3)
            .repeat(self.n_joints - 1, 1, 1)[:, :, :2]
            .flatten(1)
            .reshape(1, -1)
        )
        init_body_pose = torch.cat([init_root_pose, init_body_pose], -1)
        self.register_buffer("init_body_pose", init_body_pose)

        # trackfeat
        if self.trackfeat != "":
            from ...models.trackfeat import TrackFeatRegressor

            self.trackfeat_regressor = TrackFeatRegressor(
                config.trackfeat, self.decoder.dim, config.trackfeat_arch
            )

    def forward(
        self,
        z: EncoderOutput,
        K: Tensor,
        conf_thresh: float = 0.4,
        dist_thresh_nms: float = 0.25,
        lowres: bool = False,
    ) -> list[DecoderOutput]:
        """Decode encoder features into per-person body pose, shape, and position.

        Args:
            z: EncoderOutput containing image features and camera intrinsics.
            K: Camera intrinsics matrix of shape (bs, 3, 3).
            conf_thresh: Confidence threshold; queries below this are discarded.
            dist_thresh_nms: 3D pelvis NMS distance threshold applied per batch
                element.
            lowres: If True, runs the low-resolution Anny body model.

        Returns:
            List of DecoderOutput, one per element in the batch.
        """
        bs, nx, ny, D = z.feat.shape
        assert nx <= self.grid_size and ny <= self.grid_size, (
            f"input grid ({nx}, {ny}) exceeds positional-embedding grid {self.grid_size}"
        )

        # Context - KV  [bs,np,np,D]
        dec_emb = self.dec_to_token(z.feat) + self.dec_pos_emb[:, :nx, :ny, :]
        context = dec_emb.flatten(1, 2)  # [bs,np*np,D]

        # Queries - Q
        queries = self.queries.weight.unsqueeze(0).repeat(bs, 1, 1)  # [bs, N, D]

        y = self.decoder(x=queries, context=context)

        # Human primary keypoint 2D location
        loc = self.mlp_loc(y)
        loc = 1.2 * self.grid_size * self.patch_size * torch.sigmoid(loc)

        # Translation in camera space
        _dist = self.mlp_dist(y)
        dist = K[:, :1, :1] / torch.clamp(torch.exp(_dist), 1e-5)

        transl = inverse_perspective_projection(
            loc,
            K.unsqueeze(1).repeat(1, self.num_queries, 1, 1),
            dist,
        )

        # Confidence of the prediction
        conf = torch.sigmoid(self.mlp_conf(y))

        # select queries by confidence
        b_index, q_index, _ = torch.where(conf > conf_thresh)
        if b_index.numel() == 0:
            empty = lambda *shape: conf.new_empty(shape)
            return [
                DecoderOutput(
                    K=K[b],
                    persons=PersonOutput(
                        num_person=0,
                        conf=empty(0, 1),
                        shape=empty(0, self.num_betas),
                        v3d=empty(0, 0, 3),
                        j3d=empty(0, self.n_joints, 3),
                        j2d=empty(0, self.n_joints, 2),
                        transl_pelvis=empty(0, 1, 3),
                        bone_poses=empty(0, self.n_joints, 4, 4),
                        rest_bone_poses=empty(0, self.n_joints, 4, 4),
                        trackfeat=empty(0, 0) if self.trackfeat else None,
                    ),
                )
                for b in range(bs)
            ]

        # Extract the selected queries' information
        _dist = _dist[b_index, q_index]
        transl = transl[b_index, q_index]
        y = y[b_index, q_index]
        conf = conf[b_index, q_index]

        shape = torch.sigmoid(self.mlp_shape(y))

        # similar as iterative regressor
        rot6d = (
            self.mlp_pose(torch.cat([y, self.init_body_pose.repeat(y.shape[0], 1)], 1))
            + self.init_body_pose
        )
        rotmat = roma.special_gramschmidt(rot6d.reshape(-1, 3, 2))
        rotmat = rotmat.view(-1, self.n_joints, 3, 3)

        # Human parametric model
        _shape = {
            k: shape[:, l] for l, k in enumerate(self.body_model.phenotype_labels)
        }

        # rotmat to homogenous matrix
        rotmat_homo = rotation_to_homogeneous(rotmat)

        B = rotmat.shape[0]
        dtype = rotmat_homo.dtype
        local_changes_kwargs = {
            key: torch.zeros(B, dtype=dtype).to(rotmat.device)
            for key in self.body_model.local_change_labels
        }
        rotmat_homo[:, self._indices_to_discard] = torch.eye(4).to(rotmat.device)

        if lowres:
            with torch.amp.autocast("cuda", enabled=False):
                output = self.lowres_body_model(
                    pose_parameters=rotmat_homo,
                    phenotype_kwargs=_shape,
                    local_changes_kwargs=local_changes_kwargs,
                )
        else:
            output = self.body_model(
                pose_parameters=rotmat_homo,
                phenotype_kwargs=_shape,
                local_changes_kwargs=local_changes_kwargs,
            )

        v3d = output["vertices"]
        j3d = output["bone_poses"][:, :, :3, -1]

        if lowres:
            person_center = v3d[:, [self.person_center_idx_lowres]]
        else:
            person_center = v3d[:, [self.person_center_idx]]

        # Adding translation
        v3d, j3d = [x - person_center + transl.unsqueeze(1) for x in [v3d, j3d]]

        joints_dir = Path("output") / "3d_joints"
        joints_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"joints": j3d.detach().cpu()},
            joints_dir / "joints.pt",
        )
        with (joints_dir / "joints.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "joints": j3d.detach().cpu().tolist(),
                    "shape": list(j3d.shape),
                    "num_people": j3d.shape[0],
                    "num_joints": j3d.shape[1],
                    "coordinates": 3,
                },
                file,
            )

        j2d = perspective_projection(j3d, K[b_index])

        # At this stage we need to dispatch per batch
        trackfeat = self.trackfeat_regressor(y) if self.trackfeat else None
        outputs = []
        for b in range(bs):
            persons = PersonOutput(
                num_person=(b_index == b).sum().item(),
                conf=conf[b_index == b],
                shape=shape[b_index == b],
                v3d=v3d[b_index == b],
                j3d=j3d[b_index == b],
                j2d=j2d[b_index == b],
                transl_pelvis=j3d[b_index == b, [0]],  # root=pelvis,
                bone_poses=output["bone_poses"][b_index == b],
                rest_bone_poses=output["rest_bone_poses"][b_index == b],
                trackfeat=(trackfeat[b_index == b] if trackfeat is not None else None),
            )
            out = DecoderOutput(
                K=K[b], persons=nms_3d_pelvis(persons, dist_thresh=dist_thresh_nms)
            )
            outputs.append(out)

        return outputs
