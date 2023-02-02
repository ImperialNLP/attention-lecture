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

        embedded = self.embedding(inputs)
        # shape(embedded) = [B, I, E]

        zeros = torch.zeros(1, inputs.shape[0], self.forwards_rnn.hidden_size)

        forwards, _ = self.forwards_rnn(embedded, zeros)
        # shape(forwards) = [B, I, D]

        reverse_embedded = embedded.fliplr()

        backwards, _ = self.backwards_rnn(reverse_embedded, zeros)
        # shape(backwards) = [B, I, D]

        output = torch.cat((forwards, backwards), dim=-1)
        # shape(output) = [B, I, 2D]

        final_hidden = output[:, -1]
        final_hidden = self.projection(final_hidden)
        # shape(final_hidden) = [B, D]

        return output, final_hidden


encoder = Encoder(5, 10, 20)

dummy_inputs = torch.tensor([[1, 2, 3, 4, 0, 0], [4, 4, 3, 2, 0, 0], [1, 3, 3, 3, 2, 0]]
                            ).long()  # [3,6] (e.g. B=3, I=6)

output, hidden = encoder(dummy_inputs)
print(output.shape, hidden.shape)
