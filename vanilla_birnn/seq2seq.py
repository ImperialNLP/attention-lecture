import torch
import torch.nn as nn
import random
from encoder import Encoder
from decoder import Decoder


class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = trg.shape[1]

        c = self.encoder(src)
        # shape(c) = [B, D]

        # initialize output sequence with '<sos>'
        dec_output = trg[:, 0]
        hidden = c.unsqueeze(0)
        # shape(dec_output) = [B]
        # shape(hidden) = [1, B, D]

        outputs = torch.zeros(batch_size, 1)

        # decoder token by token
        for t in range(1, max_len):
            logits, hidden = self.decoder(dec_output, hidden)
            model_preds = logits.argmax(1)
            outputs = torch.cat((outputs, model_preds.unsqueeze(-1)), dim=1)

            # apply teacher forcing
            dec_output = trg[:, t] if random.random() < teacher_forcing_ratio else model_preds

        return outputs


encoder = Encoder(6, 10, 20)
decoder = Decoder(6, 10, 20)
s2s = Seq2seq(encoder, decoder)

src = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [1, 4, 4, 3, 2, 0, 0], [1, 3, 3, 3, 5, 2, 0]]).long()
trg = torch.tensor([[1, 5, 4, 3, 2, 0, 0], [1, 2, 3, 4, 4, 0, 0], [1, 2, 5, 3, 3, 3, 0]]).long()

outputs = s2s(src, trg)
print(outputs)
