import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        # > Initialise an embedding layer
        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # > Create an output layer (non-linearity followed by projection to vocabulary size)
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, trg_vocab_size)
        )

    def forward(self, prev_output, prev_hidden_state):
        # shape(prev_output) = [B]
        # shape(prev_hidden_state) = [1, B, D]
        print("prev_hidden_state", prev_hidden_state.shape)

        # > unsqueeze prev_output so we can feed it into the embedding layer
        prev_output = prev_output.unsqueeze(1)
        # shape(prev_input) = [B, 1]

        # > embed prev_output
        embedded = self.embedding(prev_output)
        # shape(embedded) = [B, 1, E]

        # > run the RNN, passing in the initial state as `prev_hidden_state`
        _, hidden = self.rnn(embedded, prev_hidden_state)
        # shape(hidden) = [1, B, D]

        # > use rnn_output to obtain your logits. Remove the redundant dimension before feeding it to the projection layer
        logits = self.out(hidden.squeeze(0))
        # shape(logits) = [B, V_trg]

        return logits, hidden
