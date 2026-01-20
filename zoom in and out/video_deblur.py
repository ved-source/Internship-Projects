import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import argparse
from pathlib import Path
import math


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """Reverse window partition."""
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(torch.nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define relative position bias table
        self.relative_position_bias_table = torch.nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        # Get pair-wise relative position index
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(torch.nn.Module):
    """Swin Transformer Block."""

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size

        x = self.norm1(x)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1] * window_size[2], C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)

        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x


class PatchEmbed3D(torch.nn.Module):
    """Video to Patch Embedding."""

    def __init__(self, patch_size=(1, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = torch.nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class BasicLayer(torch.nn.Module):
    """A basic Swin Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, window_size=(1, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=torch.nn.LayerNorm, downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # Build blocks
        self.blocks = torch.nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # Calculate attention mask for SW-MSA
        B, L, C = x.shape
        D = H = W = int(round(L ** (1 / 3)))  # Simplified assumption
        x = x.view(B, D, H, W, C)

        # Create attention mask
        img_mask = torch.zeros((1, D, H, W, 1), device=x.device)
        cnt = 0
        for d in slice(-self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(
                -self.shift_size[0], None):
            for h in slice(-self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(
                    -self.shift_size[1], None):
                for w in slice(-self.window_size[2]), slice(-self.window_size[2], -self.shift_size[2]), slice(
                        -self.shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class VRT(torch.nn.Module):
    """Video Restoration Transformer (VRT) model architecture"""

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=180,
                 depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4], num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                 window_size=[3, 8, 8], num_frames=6, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=torch.nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1.,
                 upsampler='', resi_connection='1conv', **kwargs):
        super(VRT, self).__init__()

        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.num_frames = num_frames
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=(1, patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = torch.nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            torch.nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = torch.nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer) if i_layer < self.num_layers // 2 else int(
                    embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None)
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)

        # Build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = torch.nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = torch.nn.Sequential(
                torch.nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                torch.nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                torch.nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        # Shallow feature extraction
        self.conv_first = torch.nn.Conv3d(in_chans, embed_dim, 3, 1, 1)

        # High quality image reconstruction
        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = torch.nn.Sequential(
                torch.nn.Conv3d(embed_dim, num_feat, 3, 1, 1), torch.nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = torch.nn.Conv3d(num_feat, out_chans, 3, 1, 1)
        else:
            # For most video restoration tasks, we don't need upsampling
            self.conv_last = torch.nn.Conv3d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3], x.shape[4])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x, _ = layer(x)

        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, x_size[0], x_size[1], x_size[2])
        return x

    def forward(self, x):
        # Shallow feature extraction
        x = self.conv_first(x)
        x_res = x.clone()

        # Deep feature extraction
        x = self.forward_features(x)
        x = self.conv_after_body(x) + x_res

        # High quality image reconstruction
        x = self.conv_last(x)

        return x


def load_vrt_model(model_path, device='cuda'):
    """Load VRT model from .pth file"""
    try:
        # Load the state dict
        checkpoint = torch.load(model_path, map_location=device)

        # Initialize model with proper parameters
        model = VRT(
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=180,
            depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            window_size=[3, 8, 8],
            num_frames=6,
            mlp_ratio=2.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            upscale=1,
            img_range=1.0,
            upsampler='',
            resi_connection='1conv'
        )

        # Load weights with proper key handling
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle potential key mismatches
        model_state_dict = model.state_dict()
        filtered_state_dict = {}

        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping key {k} due to shape mismatch or missing key")

        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        model.to(device)
        model.eval()

        print(f"✓ Model loaded successfully from {model_path}")
        print(f"✓ Loaded {len(filtered_state_dict)}/{len(model_state_dict)} parameters")
        return model

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(
            "This might be due to architecture mismatch. You may need the exact VRT implementation used for training.")
        return None


def preprocess_frames(frames, target_size=(256, 256)):
    """Preprocess video frames for VRT input"""
    processed_frames = []

    for frame in frames:
        # Resize frame
        frame = cv2.resize(frame, target_size)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        processed_frames.append(frame)

    return np.array(processed_frames)


def postprocess_frames(frames):
    """Postprocess VRT output frames"""
    processed_frames = []

    for frame in frames:
        # Denormalize
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        processed_frames.append(frame)

    return processed_frames


def extract_frames(video_path, max_frames=None):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    print(f"✓ Extracted {len(frames)} frames from video")
    return frames


def save_video(frames, output_path, fps=30):
    """Save frames as video"""
    if not frames:
        print("✗ No frames to save")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"✓ Video saved to {output_path}")


def deblur_video_batch(model, frames, device='cuda', batch_size=6):
    """Deblur video frames using VRT model"""
    model.eval()
    deblurred_frames = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            # Get actual frames for this batch
            actual_frames = min(len(frames) - i, batch_size)
            batch_frames = frames[i:i + actual_frames]

            # Convert to list if it's a numpy slice
            if isinstance(batch_frames, np.ndarray):
                batch_frames = [batch_frames[j] for j in range(len(batch_frames))]

            # Pad batch if necessary
            while len(batch_frames) < batch_size:
                batch_frames.append(batch_frames[-1].copy())

            # Convert to tensor: (T, H, W, C) -> (B, C, T, H, W)
            batch_tensor = torch.from_numpy(np.array(batch_frames)).float()  # (T, H, W, C)
            batch_tensor = batch_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
            batch_tensor = batch_tensor.unsqueeze(0)  # (B, C, T, H, W)
            batch_tensor = batch_tensor.to(device)

            print(f"Input tensor shape: {batch_tensor.shape}")  # Debug info

            try:
                # Forward pass
                output = model(batch_tensor)
                print(f"Output tensor shape: {output.shape}")  # Debug info

                # Convert back to numpy: (B, C, T, H, W) -> (T, H, W, C)
                output = output.squeeze(0)  # (C, T, H, W)
                output = output.permute(1, 2, 3, 0)  # (T, H, W, C)
                output = output.cpu().numpy()

                # Take only the original number of frames
                deblurred_frames.extend([output[j] for j in range(actual_frames)])

            except Exception as e:
                print(f"✗ Error processing batch {i // batch_size + 1}: {e}")
                # Fallback: return original frames
                for j in range(actual_frames):
                    if isinstance(batch_frames[j], np.ndarray):
                        deblurred_frames.append(batch_frames[j])
                    else:
                        deblurred_frames.append(frames[i + j])

    return deblurred_frames


def main():
    parser = argparse.ArgumentParser(description='Video Deblurring with VRT')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to VRT model (.pth file)')
    parser.add_argument('--input_video', type=str, required=True,
                        help='Path to input blurry video')
    parser.add_argument('--output_video', type=str, required=True,
                        help='Path to output deblurred video')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size for processing')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"✗ Model file not found: {args.model_path}")
        return

    if not os.path.exists(args.input_video):
        print(f"✗ Input video not found: {args.input_video}")
        return

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading VRT model...")
    model = load_vrt_model(args.model_path, device)

    if model is None:
        return

    # Extract frames
    print("Extracting frames from video...")
    frames = extract_frames(args.input_video, args.max_frames)

    if not frames:
        print("✗ No frames extracted from video")
        return

    # Preprocess frames
    print("Preprocessing frames...")
    processed_frames = preprocess_frames(frames)

    # Deblur frames
    print("Deblurring frames...")
    deblurred_frames = deblur_video_batch(model, processed_frames, device, args.batch_size)

    # Postprocess frames
    print("Postprocessing frames...")
    final_frames = postprocess_frames(deblurred_frames)

    # Save output video
    print("Saving output video...")
    save_video(final_frames, args.output_video)

    print(f"✓ Video deblurring completed! Output saved to: {args.output_video}")


if __name__ == "__main__":
    # For direct execution without command line arguments
    if len(os.sys.argv) == 1:
        # Configuration for direct execution
        MODEL_PATH = r"C:\Users\saive\Downloads\006_VRT_videodeblurring_GoPro.pth"  # Replace with your model path
        INPUT_VIDEO = r"C:\Users\saive\OneDrive\Desktop\Desktop\all folders\self-projects\placement\internship\zoom in and out\WhatsApp Video 2025-06-02 at 14.39.39_074daeb3.mp4"  # Replace with your input video
        OUTPUT_VIDEO = "deblurred_video.mp4"  # Replace with desired output path

        print("=" * 60)
        print("VRT Video Deblurring Script")
        print("=" * 60)

        # Check CUDA availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load model
        print("\n1. Loading VRT model...")
        model = load_vrt_model(MODEL_PATH, device)

        if model is None:
            print("Please check your model path and try again.")
            exit(1)

        # Extract frames
        print("\n2. Extracting frames from video...")
        frames = extract_frames(INPUT_VIDEO, max_frames=100)  # Limit to 100 frames for testing

        if not frames:
            print("No frames extracted. Please check your input video path.")
            exit(1)

        # Preprocess frames
        print("\n3. Preprocessing frames...")
        processed_frames = preprocess_frames(frames)

        # Deblur frames
        print("\n4. Deblurring frames...")
        deblurred_frames = deblur_video_batch(model, processed_frames, device, batch_size=6)

        # Postprocess frames
        print("\n5. Postprocessing frames...")
        final_frames = postprocess_frames(deblurred_frames)

        # Save output video
        print("\n6. Saving output video...")
        save_video(final_frames, OUTPUT_VIDEO)

        print(f"\n✓ Video deblurring completed!")
        print(f"Output saved to: {OUTPUT_VIDEO}")
        print("=" * 60)
    else:
        main()