"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMUTransformer2heads(nn.Module):

    def __init__(self, num_inputs, num_outputs, config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = config.get("transformer_dim")
        #self.input_proj = nn.Conv1d(num_inputs, self.transformer_dim, 1)

        self.input_proj = nn.Sequential(nn.Conv1d(num_inputs, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())


        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = config.get("nhead"),
                                       dim_feedforward = config.get("dim_feedforward"),
                                       dropout = config.get("transformer_dropout"),
                                       activation = config.get("transformer_activation"))

        self.transformer_encoder1 = TransformerEncoder(encoder_layer,
                                              num_layers = config.get("num_encoder_layers"),
                                              norm = nn.LayerNorm(self.transformer_dim))

        self.transformer_encoder2 = TransformerEncoder(encoder_layer,
                                              num_layers = config.get("num_encoder_layers"),
                                              norm = nn.LayerNorm(self.transformer_dim))

        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        self.imu_head1 = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  num_outputs)
        )
        self.imu_head2 = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  num_outputs)
        )
        # self.imu_head  = nn.Sequential(
        #     nn.Linear(in_planes * in_dim, fc_dim),
        #     nn.ReLU(True),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_dim, fc_dim),
        #     nn.ReLU(True),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_dim, num_outputs))

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, y=None):
        src = x  # Shape N x S x C with S = sequence length, N = batch size, C = channels

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        target1 = self.transformer_encoder1(src)[0]
        target2 = self.transformer_encoder2(src)[0]

        # Class probability
        target1 = self.imu_head1(target1)
        target2 = self.imu_head2(target2)
        target = torch.cat([target1, target2], axis=1)
        return target

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))


