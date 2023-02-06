import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # > Create an output layer (tanh non-linearity followed by projection to vocabulary size)
        self.out = nn.Sequential(


        )

    def forward(self, prev_output, prev_hidden_state):
        # shape(prev_output) = [B]
        # shape(prev_hidden_state) = [1, B, D]

        # > unsqueeze prev_output so we can feed it into the embedding layer
        prev_output =
        # shape(prev_input) = [B, 1]

        # > embed prev_output
        embedded =
        # shape(embedded) = [B, 1, E]

        # > run the RNN, passing in the initial state as `prev_hidden_state`
        _, hidden =
        # shape(hidden) = [1, B, D]

        # > use rnn_output to obtain your logits. Remove the redundant dimension before feeding it to the projection layer
        logits =
        # shape(logits) = [B, V_trg]

        return logits, hidden
