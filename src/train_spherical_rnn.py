import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

# 1) Модель: простая RNN (LSTM) с выходом-эмбеддингом размерности D
class SphericalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        _, (h_n, _) = self.lstm(x)           # h_n: [n_layers, batch, hidden_size]
        h_last      = h_n[-1]                # [batch, hidden_size]
        z            = self.fc(h_last)       # [batch, embedding_dim]
        z_sphere     = nn.functional.normalize(z, p=2, dim=1)  # L2-нормализация
        return z_sphere                       # все векторы на S^{D-1}

# 2) Функция тренировки с мини-батч кластеризацией  
def train_epoch(model, dataloader, optimizer, device, n_clusters=10):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch.to(device)                # [batch, seq_len, input_size]
        z = model(x)                        # [batch, D] — на сфере

        # 2.1) Кластеризуем эмбеддинги (mini-batch k-means на сфере)
        #      Используем косинусное расстояние = 1 - dot(z_i, center_j)
        #      sklearn.KMeans по умолчанию — Евклид, но на L2-нормированных данных
        #      Евклидова метрика эквивалентна косинусу.
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(z.detach().cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, device=device)
        centers = nn.functional.normalize(centers, p=2, dim=1)  # центры тоже на сфере

        # 2.2) Loss: среднее косинус-расстояние до своего центра
        #      косинус-расстояние = 1 - dot(z, center)
        z_expand      = z.unsqueeze(1)                   # [batch, 1, D]
        centers_expand = centers.unsqueeze(0)            # [1, n_clusters, D]
        # dot: [batch, n_clusters]
        cos_sim      = torch.bmm(z_expand, centers_expand.transpose(1,2)).squeeze(1)
        # собираем по меткам
        chosen_sim   = cos_sim[torch.arange(z.size(0)), labels]
        loss         = torch.mean(1.0 - chosen_sim)      # хотим максимум сходства → минимум (1 - cos)

        # 2.3) Шаг оптимизации
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * z.size(0)

    return total_loss / len(dataloader.dataset)

# 3) Пример использования
if __name__ == "__main__":
    # Параметры
    input_size   = 5       # размерность вашего признакового вектора на шаге time-series
    hidden_size  = 64
    embedding_dim= 16      # размерность сферы S^{15}
    n_clusters   = 8
    lr           = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SphericalRNN(input_size, hidden_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Датасет: ваш DataLoader для time-series
    # должен отдавать тензор shape=[batch, seq_len, input_size]
    from torch.utils.data import DataLoader, TensorDataset
    # пример: случайные данные
    dummy = torch.randn(1000, 50, input_size)
    loader = DataLoader(TensorDataset(dummy), batch_size=32, shuffle=True)

    # Тренировка
    for epoch in range(1, 11):
        avg_loss = train_epoch(model, loader, optimizer, device, n_clusters)
        print(f"Epoch {epoch:02d}, Loss = {avg_loss:.4f}")