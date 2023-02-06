import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        # NOTE: We _could_ initialise a separate W and U to project our inputs.
        # However, we can also utilise ONE set of weights that projects both inputs at once.
        # We simply concatenate our desired inputs together and then feed them to our layer.
        # > With that said, create the alignment function
        self.alignment_layer = nn.Sequential(


        )

        # > initialise a V layer (set bias to False)
        self.v =

    def forward(self, prev_dec_hidden, enc_outputs):
        # shape(dec_hidden) = [1, B, D]
        # shape(enc_outputs) = [B, I, 2D]

        # > prev_dec_hidden should be [B, 1, D]
        prev_dec_hidden =
        # shape(dec_hidden) = [B, 1, D]

        # > tile/repeat prev_dec_hidden for each word in your source
        src_len = enc_outputs.shape[1]  # I
        prev_dec_hidden =
        # shape(prev_dec_hidden) = [B, I, D]

        # > concatenate previous decoder hidden sates and encoder states
        concatenated =
        # shape(concatenated) = [B, I, 3D]
        energy =
        # shape(energy) = [B, I, D]

        # > obtain your attention scores/alpha
        attention =
        # shape(attention) = [B, I]

        return attention
