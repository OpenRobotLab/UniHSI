# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from abc import ABCMeta, abstractmethod, abstractclassmethod
from collections import OrderedDict
import json

import numpy as np
import os
from ..rotation3d import *

# from poselib.poselib.skeleton.skeleton3d import SkeletonMotion


TENSOR_CLASS = {}


def register(name):
    global TENSOR_CLASS

    def core(tensor_cls):
        TENSOR_CLASS[name] = tensor_cls
        return tensor_cls

    return core


def _get_cls(name):
    global TENSOR_CLASS
    return TENSOR_CLASS[name]


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return dict(__ndarray__=obj.tolist(), dtype=str(obj.dtype), shape=obj.shape)
        return json.JSONEncoder.default(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and "__ndarray__" in dct:
        data = np.asarray(dct["__ndarray__"], dtype=dct["dtype"])
        return data.reshape(dct["shape"])
    return dct


class Serializable:
    """ Implementation to read/write to file.
    All class the is inherited from this class needs to implement to_dict() and 
    from_dict()
    """

    @abstractclassmethod
    def from_dict(cls, dict_repr, *args, **kwargs):
        """ Read the object from an ordered dictionary

        :param dict_repr: the ordered dictionary that is used to construct the object
        :type dict_repr: OrderedDict
        :param args, kwargs: the arguments that need to be passed into from_dict()
        :type args, kwargs: additional arguments
        """
        pass

    @abstractmethod
    def to_dict(self):
        """ Construct an ordered dictionary from the object
        
        :rtype: OrderedDict
        """
        pass

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        """ Read the object from a file (either .npy or .json)

        :param path: path of the file
        :type path: string
        :param args, kwargs: the arguments that need to be passed into from_dict()
        :type args, kwargs: additional arguments
        """
        if path.endswith(".json"):
            with open(path, "r") as f:
                d = json.load(f, object_hook=json_numpy_obj_hook)
        elif path.endswith(".npy"):
            d = np.load(path, allow_pickle=True).item()
        else:
            assert False, "failed to load {} from {}".format(cls.__name__, path)
        assert d["__name__"] == cls.__name__, "the file belongs to {}, not {}".format(
            d["__name__"], cls.__name__
        )
        # hack
        # d['rotation']['arr'][:,0,0:2] *= np.sqrt(d['rotation']['arr'][:,0,2]*d['rotation']['arr'][:,0,2] / 
        #                                                           (d['rotation']['arr'][:,0,0]*d['rotation']['arr'][:,0,0] + d['rotation']['arr'][:,0,1]*d['rotation']['arr'][:,0,1]) + 1)[..., None]

        # d['rotation']['arr'][:,0,2] = 0.0

        # from scipy.spatial.transform import Rotation as R
        
        # def quaternion2euler(quaternion):
        #     r = R.from_quat(quaternion)
        #     euler = r.as_euler('zyx', degrees=True)
        #     return euler

        # def euler2quaternion(euler):
        #     r = R.from_euler('zyx', euler, degrees=True)
        #     quaternion = r.as_quat()
        #     return quaternion

        # for i in range(len(d['rotation']['arr'])):
        #     q = d['rotation']['arr'][i,0]
        #     v = torch.tensor([1, 0, 0])
        #     v_r = quat_rotate(torch.from_numpy(q), v)
        #     phi = torch.arctan2(v_r[1], v_r[0])
        #     q_r = torch.tensor([0,0,np.sin(-phi/2), np.cos(-phi/2)])
        #     q = quat_mul(q_r, torch.from_numpy(q)).numpy()
        #     d['rotation']['arr'][i,0] = q
        # for i in range(len(d['rotation']['arr'])):
        #     q = d['rotation']['arr'][i,0]
        #     q_r = torch.tensor([0,0,np.sin(-np.pi/2), np.cos(-np.pi/2)])
        #     q = quat_mul(q_r, torch.from_numpy(q)).numpy()
        #     d['rotation']['arr'][:,0] = q

        # min_index = d['root_translation']['arr'][:, 2].argmin()
        # x = d['root_translation']['arr'][min_index, 0]
        # y = d['root_translation']['arr'][min_index, 1]
        # q = d['rotation']['arr'][min_index,0]
        # v = torch.tensor([1, 0, 0])
        # v_r = quat_rotate(torch.from_numpy(q), v)
        # phi = torch.atan2(v_r[1], v_r[0])
        # q_r = torch.tensor([0,0,np.sin(-phi/2), np.cos(-phi/2)])
        # for i in range(len(d['rotation']['arr'])):
        #     q = d['rotation']['arr'][i,0]
        #     q = quat_mul(q_r, torch.from_numpy(q)).numpy()
        #     d['rotation']['arr'][i,0] = q
        #     d['root_translation']['arr'][i, 0] -= x
        #     d['root_translation']['arr'][i, 1] -= y
        #     v_r = quat_rotate(q_r, torch.from_numpy(d['root_translation']['arr'][i]))
        #     d['root_translation']['arr'][i] = v_r.numpy()

        if 'target_pos' in d:
            return cls.from_dict(d, *args, **kwargs), d['target_pos']
        return cls.from_dict(d, *args, **kwargs), None

    def to_file(self, path: str) -> None:
        """ Write the object to a file (either .npy or .json)

        :param path: path of the file
        :type path: string
        """
        if os.path.dirname(path) != "" and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        d = self.to_dict()
        d["__name__"] = self.__class__.__name__
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(d, f, cls=NumpyEncoder, indent=4)
        elif path.endswith(".npy"):
            np.save(path, d)
