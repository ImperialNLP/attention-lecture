import torch
import torch.nn as nn

from attention import Attention


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim)

        self.attention = Attention(hidden_dim)

        # > initialise the RNN
        self.rnn = nn.RNN(embedding_dim + 2 * hidden_dim, hidden_dim, batch_first=True)

        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(3 * hidden_dim + embedding_dim, trg_vocab_size)
        )

    def forward(self, prev_output, prev_hidden_state, all_encoder_states):
        # shape(prev_output) = [B]
        # shape(prev_hidden_state) = [1, B, D]
        # shape(all_encoder_states) = [B, I, 2D]

        prev_output = prev_output.unsqueeze(1)
        # shape(prev_output) = [B, 1]

        embedded = self.embedding(prev_output)
        # shape(embedded) = [B, 1, E]

        # > Obtain your alphas
        alphas = self.attention(prev_hidden_state, all_encoder_states)
        # shape(alphas) = [B, I]

        # > alphas should be [B, 1, I]
        alphas = alphas.unsqueeze(1)
        # shape(alphas) = [B, 1, I]

        # > apply your alphas to the encoder hidden states
        c = torch.bmm(alphas, all_encoder_states)
        # shape(c) = [B, 1, 2D]

        # > concatenate your context vector with the input embedding
        input_for_rnn = torch.cat((embedded, c), dim=2)
        # shape(input_for_rnn) = [B, 1, E+2D]

        # > run the RNN, passing in the initial state as `prev_hidden_state`
        rnn_outputs, hidden = self.rnn(input_for_rnn, prev_hidden_state)
        # shape(rnn_outputs) = [B, 1, D]
        # shape(hidden) = [1, B, D]

        # > concatenate the word embedding, rnn_outputs and c
        output = torch.cat((embedded, rnn_outputs, c), dim=2)
        # shape(output) = [B, 1, E + 3D]

        # > Project the output and remove the redundant dimension
        logits = self.out(output.squeeze(1))
        # shape(logits) = [B, V_trg]

        return logits, hidden, alphas.squeeze(1)
