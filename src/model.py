from torch import nn
import torch.nn.functional as F
import math


class SphericalRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        embedding_dim: int,
        n_layers: int = 1,
        mlp_hidden_dims: list[int] = (64, 32),
    ):
        """
        input_size:      размер входного вектора на каждом шаге
        hidden_size:     число юнитов в LSTM
        embedding_dim:   выходная размерность (сфера S^{embedding_dim-1})
        n_layers:        число слоёв LSTM
        mlp_hidden_dims: кортеж с размерами скрытых слоёв perceptron'а
        """
        super().__init__()
        # --- RNN part ---
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=n_layers, batch_first=True
        )

        # --- Perceptron (MLP) after RNN ---
        mlp_layers: list[nn.Module] = []
        in_dim = hidden_size
        for hdim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, hdim))
            mlp_layers.append(nn.SELU())
            in_dim = hdim
        # final projection to embedding_dim
        mlp_layers.append(nn.Linear(in_dim, embedding_dim))
        self.mlp = nn.Sequential(*mlp_layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # LeCun Normal for SELU nets: std = 1/√fan_in
        if isinstance(m, (nn.Linear,)):
            fan_in = m.weight.size(1)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # For LSTM: use LeCun normal on input→hidden, orthogonal on hidden→hidden
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    fan_in = m.input_size
                    nn.init.normal_(param, 0.0, 1.0 / math.sqrt(fan_in))
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        _, (h_n, _) = self.lstm(x)  # h_n: [n_layers, batch, hidden_size]
        h_last = h_n[-1]  # [batch, hidden_size]
        z = self.mlp(h_last)  # [batch, embedding_dim], via tanh MLP
        z_sphere = F.normalize(z, p=2, dim=1)  # проектируем на единичную сферу
        return z_sphere
