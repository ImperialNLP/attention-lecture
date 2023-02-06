import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.forwards_rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.backwards_rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, inputs):
        # shape(inputs) = [B, I]

        # > embed the inputs
        embedded =
        # shape(embedded) = [B, I, E]

        zeros = torch.zeros(1, inputs.shape[0], self.forwards_rnn.hidden_size)

        # > run the forwards RNN
        _, final_forwards =
        # shape(final_forwards) = [1, B, D]

        # > reverse the sequence to feed into the backwards rnn (hint: fliplr)
        # > why do we use fliplr instead of reverse?
        reverse_embedded =

        # > run the backwards RNN
        _, final_backwards =
        # shape(final_backwards) = [1, B, D]

        # > ensure that final_forward and final_backwards are [B, D]
        final_forwards =
        final_backwards =

        # > concatenate final_forward and final_backwards
        output =
        # shape(output) = [B, 2D]

        # > project output back to D dimensionality
        output =
        # shape(output) = [B, D]

        return output


encoder = Encoder(5, 10, 20)

dummy_inputs = torch.tensor([[1, 2, 3, 4, 0, 0], [4, 4, 3, 2, 0, 0], [1, 3, 3, 3, 2, 0]]
                            ).long()  # [3,6] (e.g. B=3, I=6)

print(encoder(dummy_inputs).shape)
