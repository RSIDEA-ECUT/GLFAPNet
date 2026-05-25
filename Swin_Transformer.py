import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    动态窗口划分，支持任意大小输入
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    动态窗口合并，支持任意大小输入
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    修改后的窗口注意力模块，支持动态输入大小
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表现在使用可学习的插值
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        支持动态输入大小的前向传播
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 动态计算相对位置偏置
        Wh, Ww = int(N ** 0.5), int(N ** 0.5)  # 假设输入是方形窗口
        relative_position_bias = self._get_relative_pos_bias(Wh, Ww)
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

    def _get_relative_pos_bias(self, H, W):
        """动态生成相对位置偏置"""
        # 生成相对位置索引
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, H, W
        coords_flatten = torch.flatten(coords, 1)  # 2, H*W
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2

        # 转换为非负索引
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # H*W, H*W

        # 从表中获取偏置
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(H * W, H * W, -1)  # H*W, H*W, nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, H*W, H*W


# 保持DropPath和Mlp实现不变
class DropPath(nn.Module):
    """随机深度dropout"""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP模块"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    支持任意输入大小的Swin Transformer块
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = (window_size, window_size)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 确保shift_size小于window_size
        if isinstance(shift_size, (list, tuple)):
            assert len(shift_size) == 2
            self.shift_size = shift_size
        else:
            self.shift_size = (shift_size, shift_size)

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size必须在0到window_size之间"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size必须在0到window_size之间"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        """
        支持任意输入大小的前向传播
        """
        B, C, H, W = x.shape

        # 转换为序列形式 (B, L, C)
        shortcut = x.permute(0, 2, 3, 1).view(B, H * W, C)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        # 计算填充以确保窗口划分
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # 循环移位
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # 计算注意力mask（仅当有移位时）
        if any(s > 0 for s in self.shift_size):
            # 生成mask
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # 逆循环移位
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        # 移除填充
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # 残差连接
        x = shortcut + self.drop_path(x)

        # MLP部分
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 恢复形状 (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x


# 由多个Swin Transformer(一般为偶数个)块组成BasicLayer
class BasicLayer(nn.Module):
    def __init__(self, dim=32, depth=2, num_heads=4, window_size=8):
        super(BasicLayer, self).__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


if __name__ == "__main__":
    # 创建模块
    dim = 32
    num_heads = 4
    window_size = 8
    shift_size = window_size // 2

    swin_block = SwinTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size
    )
    x_test = torch.randn(1, 32, 256, 256)
    output = swin_block(x_test)
    print("输入形状:", x_test.shape)
    print("输出形状:", output.shape)
