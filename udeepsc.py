import math
import torch
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from torch import nn
from transformers import BertModel


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=32, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.unpatch = PatchUnembedding(32, patch_size, in_chans, embed_dim)
        num_patches_imgr = self.patch_embed.num_patches
        # TODO: Add the cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_imgr = nn.Parameter(torch.zeros(1, num_patches_imgr + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.task_embedd, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        batch_size = x.shape[0]
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(x.device)
        # task_embedd = self.task_embedd.expand(batch_size, -1, -1).to(x.device)
        # x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class PatchUnembedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchUnembedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = x.reshape(-1, self.embed_dim, self.grid_size, self.grid_size)  # (B, embed_dim, grid_size, grid_size)
        x = self.proj(x)  # (B, in_chans, img_size, img_size)
        return x


class AWGNChannel(nn.Module):
    def __init__(self, snr: float = 12):
        super(AWGNChannel, self).__init__()
        self.snr = snr
        self.snr_factor = 10 ** (self.snr / 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the power of the input signal
        x_power = torch.mean(x ** 2)

        # Calculate the noise power based on SNR
        n_power = x_power / self.snr_factor

        # Generate Gaussian noise with the calculated noise power
        noise = torch.randn_like(x) * torch.sqrt(n_power)

        return x + noise

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))  # math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, policy=None, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        # print(key.shape)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        x, self.attn = self.attention(query, key, value, policy=policy, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, policy=None, mask=None, eps=1e-6):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # print(mask.shape)
        if mask is not None:
            scores += (mask * -1e9)
            # attention weights
        if policy is None:
            p_attn = F.softmax(scores, dim=-1)
            return torch.matmul(p_attn, value), p_attn
        else:
            B, N1, _ = policy.size()
            B, H, N1, N2 = scores.size()
            attn_policy = policy.reshape(B, 1, 1, N2)
            temp = torch.zeros((B, 1, N1, N2), dtype=attn_policy.dtype, device=attn_policy.device)
            attn_policy = attn_policy + temp
            max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
            scores = scores - max_scores
            scores = scores.to(torch.float32).exp_() * attn_policy.to(torch.float32)
            p_attn = (scores + eps / N1) / (scores.sum(dim=-1, keepdim=True) + eps)
            return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        # self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, policy, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        attn_output = self.self_mha(x, x, x, None, look_ahead_mask)
        x = self.layernorm1(x + attn_output)

        src_output = self.src_mha(x, memory, memory, policy, trg_padding_mask)  # q, k, v
        x = self.layernorm2(x + src_output)

        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Decoder(nn.Module):
    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate)
                                         for _ in range(depth)])

    def forward(self, x, memory, policy=None, look_ahead_mask=None, trg_padding_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, policy, look_ahead_mask, trg_padding_mask)
        return x


class UDeepSC(nn.Module):
    def __init__(self, mode="mini", patch_size=16, encoder_in_chans=3, img_embed_dim=192,
                 img_encoder_depth=4, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4,
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., channel=AWGNChannel,
                 norm_layer=nn.LayerNorm):

        super(UDeepSC, self).__init__()
        if mode == "tiny":
            self.text_embed_dim = 128
        elif mode == 'tiny':
            self.text_embed_dim = 256
        else:  # mode small
            self.text_embed_dim = 512
        self.img_embed_dim = img_embed_dim
        self.text_encoder = BertModel.from_pretrained(f"prajjwal1/bert-{mode}")
        self.vi_encoder = ViTEncoder(
            patch_size=patch_size, in_chans=encoder_in_chans, embed_dim=img_embed_dim,
            depth=img_encoder_depth, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        self.num_symbols_img = 16
        self.num_symbols_text = 16

        self.text_channel_encoder = nn.Linear(self.text_embed_dim, self.num_symbols_text)
        self.img_channel_encoder = nn.Linear(img_embed_dim, self.num_symbols_img)

        self.channel_cls = channel
        self.noise_channel = channel()

        self.text_channel_decoder = nn.Linear(self.num_symbols_text, self.text_embed_dim)
        self.img_channel_decoder = nn.Linear(self.num_symbols_img, img_embed_dim)

        self.text_decoder = Decoder(depth=decoder_depth, embed_dim=self.text_embed_dim, num_heads=decoder_num_heads)
        self.img_decoder = Decoder(depth=decoder_depth, embed_dim=img_embed_dim, num_heads=decoder_num_heads)
        self.unpatch = PatchUnembedding(32, patch_size, encoder_in_chans, img_embed_dim)
        self.text_head = nn.Sequential(
            nn.Linear(self.text_embed_dim, 30522),
            nn.Softmax(dim=2)  # dim=2 means that it calculates softmax in the feature dimension
        )

    def forward(self, x_text, x_img=None, multi_modal=False):
        attention_mask = (x_text != 0).long()
        if multi_modal:
            x_text = self.text_encoder(input_ids=x_text, attention_mask=attention_mask, return_dict=False)[0]
            x_text = self.text_channel_encoder(x_text)
            noisy_x_text = self.noise_channel(x_text)
            x_text = self.text_channel_decoder(noisy_x_text)
            x_text = self.text_decoder(x_text, x_text, None, None, None)
            x_text = self.text_head(x_text)

            x_img = self.vi_encoder(x_img)
            x_img = self.img_channel_encoder(x_img)
            noisy_x_img = self.noise_channel(x_img)
            x_img = self.img_channel_decoder(noisy_x_img)
            x_img = self.img_decoder(x_img, x_img, None, None, None)
            x_img = self.unpatch(x_img)
        else:
            x_text = self.text_encoder(input_ids=x_text, attention_mask=attention_mask, return_dict=False)[0]
            x_text = self.text_channel_encoder(x_text)
            noisy_x_text = self.noise_channel(x_text)
            x_text = self.text_channel_decoder(noisy_x_text)
            x_text = self.text_decoder(x_text, x_text, None, None, None)
            x_text = self.text_head(x_text)
            x_img = None
        return x_text, x_img
