"""Small participant-level temporal head over frozen Mouse2Vec windows."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalHead(nn.Module):
    """Learned compression within trials, then across the 20 ordered trials."""

    def __init__(self, hidden_size: int = 8, use_demographics: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_demographics = use_demographics
        self.trial_gru = nn.GRU(input_size=128, hidden_size=hidden_size, batch_first=True)
        self.participant_gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(hidden_size + (3 if use_demographics else 0), 1)

    def forward(self, windows: torch.Tensor, lengths: torch.Tensor, demographics: torch.Tensor | None = None):
        batch, n_trials, n_steps, embedding_dim = windows.shape
        if n_trials != 20 or embedding_dim != 128 or torch.any(lengths <= 0):
            raise ValueError("Expected 20 nonempty trials with 128-dimensional window embeddings")
        packed = pack_padded_sequence(
            windows.reshape(batch * n_trials, n_steps, embedding_dim),
            lengths.reshape(-1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, trial_last = self.trial_gru(packed)
        trial_states = trial_last[-1].reshape(batch, n_trials, self.hidden_size)
        _, participant_last = self.participant_gru(self.dropout(trial_states))
        participant_state = self.dropout(participant_last[-1])
        if self.use_demographics:
            if demographics is None or demographics.shape != (batch, 3):
                raise ValueError("Combined model needs three demographic values per participant")
            participant_state = torch.cat((participant_state, demographics), dim=1)
        return self.classifier(participant_state).squeeze(1)
