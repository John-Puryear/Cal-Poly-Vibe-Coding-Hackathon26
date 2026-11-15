"""Synthetic offline anatomy targets in the organism's own coordinates.

This module is for training and audit only. ProtocolNCA.forward never imports
it or receives the generated fields. The geometries are modeling assumptions,
not measurements of treated animals.
"""
import math

import numpy as np
import torch

from .data import gaussian


def intrinsic_grid(example):
    body = example.target
    size = body.shape[-1]
    info = example.landmarks
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    angle = info["angle"]
    u = ((x-info["cx"])*math.cos(angle)+(y-info["cy"])*math.sin(angle))/info["scale_u"]
    v = (-(x-info["cx"])*math.sin(angle)+(y-info["cy"])*math.cos(angle))/info["scale_v"]
    return u, v


def planaria_targets(example, eye_sigma=.05, posterior_eye_v=.11,
                    posterior_eye_u=.45):
    """Return normal and two-head targets with an identical physical body.

    The posterior eyes are placed within the tapered surviving body. This is
    direct synthetic phenotype supervision, now explicitly authorized for the
    demo; it does not appear in the inference transition.
    """
    if example.kind != "planaria":
        raise ValueError("Expected a planarian example")
    normal = example.target.clone()
    two = example.target.clone()
    u, v = intrinsic_grid(example)
    body = normal[0, 0].numpy()
    left_eyes = np.maximum(gaussian(u,v,-.52,-.11,eye_sigma,eye_sigma),
                           gaussian(u,v,-.52,+.11,eye_sigma,eye_sigma)) * body
    right_head = gaussian(u,v,+.49,0,.16,.20) * body
    right_eyes = np.maximum(gaussian(u,v,posterior_eye_u,-posterior_eye_v,eye_sigma,eye_sigma),
                            gaussian(u,v,posterior_eye_u,+posterior_eye_v,eye_sigma,eye_sigma)) * body
    left = torch.from_numpy(left_eyes.astype(np.float32))
    right_h = torch.from_numpy(right_head.astype(np.float32))
    right_e = torch.from_numpy(right_eyes.astype(np.float32))
    normal[0, 2] = left
    two[0, 1] = torch.maximum(normal[0, 1], right_h)
    two[0, 2] = torch.maximum(left, right_e)
    two[0, 3] = 0
    two[0, 4] = (.65*two[0, 1]-.65*two[0, 2]).clamp(-1,1)*two[0, 0]
    two[0, 5] = (two[0, 1]-.8*two[0, 2]).clamp(0,1)*two[0, 0]
    two[0, 6] = (.15*two[0, 0]+.8*two[0, 2]).clamp(0,1)*two[0, 0]
    two[0, 12] = torch.tanh(2*two[0, 4])*two[0, 0]
    two[0, 13] = normal[0, 13].abs()*two[0, 0]
    two[0, 14] = .75*two[0, 0]
    return normal, two
