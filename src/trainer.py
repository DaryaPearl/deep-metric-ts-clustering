from torch import nn
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


def train_epoch(
    model,
    x_train,
    optimizer,
    device,
    n_clusters=10,
    batch_size=32,
):
    dataloader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        z = model(batch[0].to(device))  # [batch, D] — на сфере

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
        z_expand = z.unsqueeze(1)  # [batch, 1, D]
        centers_expand = centers.unsqueeze(0)  # [1, n_clusters, D]
        # dot: [batch, n_clusters]
        cos_sim = z @ centers.t()  # [batch, n_clusters]
        chosen_sim = cos_sim[torch.arange(z.size(0)), labels]
        loss = torch.mean(1.0 - chosen_sim)
        # собираем по меткам
        chosen_sim = cos_sim[torch.arange(z.size(0)), labels]
        loss = torch.mean(
            1.0 - chosen_sim
        )  # хотим максимум сходства → минимум (1 - cos)

        # 2.3) Шаг оптимизации
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * z.size(0)

    return total_loss / len(dataloader.dataset)
