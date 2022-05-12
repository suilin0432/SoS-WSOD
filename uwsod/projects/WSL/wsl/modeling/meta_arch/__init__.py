# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNNWSL, ProposalNetworkWSL

__all__ = list(globals().keys())

