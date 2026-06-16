# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.options import Options

import kiui

try:
    from gsplat.rendering import rasterization
except ImportError:
    rasterization = None


def _fallback_rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    near_plane,
    far_plane,
    backgrounds=None,
    **kwargs,
):
    device = means.device
    dtype = means.dtype
    ones = torch.ones(means.shape[0], 1, device=device, dtype=dtype)
    means_h = torch.cat([means, ones], dim=-1)

    rendered_images = []
    rendered_alphas = []
    for view_idx in range(viewmats.shape[0]):
        cam = means_h @ viewmats[view_idx].to(device=device, dtype=dtype).T
        z = cam[:, 2]
        valid = (z > near_plane) & (z < far_plane)
        z_safe = z.clamp_min(1e-4)

        K = Ks[view_idx].to(device=device, dtype=dtype)
        px = cam[:, 0] / z_safe * K[0, 0] + K[0, 2]
        py = cam[:, 1] / z_safe * K[1, 1] + K[1, 2]
        ix = px.round().long()
        iy = py.round().long()
        valid = valid & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)

        image_flat = torch.zeros(height * width, 3, device=device, dtype=torch.float32)
        alpha_flat = torch.zeros(height * width, 1, device=device, dtype=torch.float32)
        if valid.any():
            idx = (iy[valid] * width + ix[valid]).long()
            alpha = opacities[valid].float().unsqueeze(-1).clamp(0, 1)
            color = colors[valid].float().clamp(0, 1)
            image_flat.scatter_add_(0, idx[:, None].expand(-1, 3), color * alpha)
            alpha_flat.scatter_add_(0, idx[:, None], alpha)

        alpha_flat = alpha_flat.clamp(0, 1)
        image_flat = image_flat / alpha_flat.clamp_min(1e-6)
        if backgrounds is not None:
            bg = backgrounds[view_idx].to(device=device, dtype=torch.float32)
            image_flat = image_flat * alpha_flat + bg.view(1, 3) * (1 - alpha_flat)

        rendered_images.append(image_flat.view(height, width, 3))
        rendered_alphas.append(alpha_flat.view(height, width, 1))

    return torch.stack(rendered_images, dim=0), torch.stack(rendered_alphas, dim=0), {"fallback": True}

class GaussianRenderer:
    def __init__(self, opt: Options):
        
        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32)
        
        # intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1

        f = self.opt.output_size / (2 * self.tan_half_fov)
        self.K = torch.tensor([[f, 0., self.opt.output_size/2.], [0., f, self.opt.output_size/2.], [0., 0., 1.]], dtype=torch.float32)

    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        device = gaussians.device
        B, V = cam_view.shape[:2]
        K = self.K.to(device)
        bg = self.bg_color.to(device) if bg_color is None else bg_color.to(device)

        # loop of loop...
        images = []
        alphas = []
        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]
                
            # render novel views
            view_matrix = cam_view[b].float()
            view_proj_matrix = cam_view_proj[b].float()
            campos = cam_pos[b].float()

            viewmat = view_matrix.transpose(2, 1) # [V, 4, 4]

            
            rasterizer = rasterization or _fallback_rasterization
            rendered_image_all, rendered_alpha_all, info = rasterizer(
                means=means3D,
                quats=rotations,
                scales=scales,
                opacities=opacity.squeeze(-1),
                colors=rgbs,
                viewmats=viewmat,
                Ks=torch.stack([K for _ in range(V)]),
                width=self.opt.output_size,
                height=self.opt.output_size,
                near_plane=self.opt.znear,
                far_plane=self.opt.zfar,
                packed=False,
                backgrounds=torch.stack([bg for _ in range(V)]),
                render_mode="RGB",
            )
            for rendered_image, rendered_alpha in zip(rendered_image_all, rendered_alpha_all):

                rendered_image = rendered_image.permute(2, 0, 1)
                rendered_image = rendered_image.clamp(0, 1)

                rendered_alpha = rendered_alpha.permute(2, 0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
