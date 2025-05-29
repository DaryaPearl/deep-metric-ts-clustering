# deep-metric-ts-clustering
End-to-end clustering of time-series data using neural networks with metric learning. This is the main repository, that was covered by the presentation. In the "dev" branch you can check our other atempt to implement stock classification using metric learning.

# Clustering Time-Series Data via End-to-End Neural Networks with Metric Learning

This project explores an end-to-end deep clustering architecture for time-series data using metric learning.

## 🔍 Project Goal

- Encode time series into a latent space with sequence models (e.g., dilated RNNs, temporal convolutional autoencoders)
- Learn a similarity metric (e.g., triplet/contrastive loss)
- Apply differentiable clustering directly on learned embeddings

## 📈 Use Case

We aim to test our method on real-world time-series datasets (e.g., stock prices, ECG signals) to demonstrate its ability to group similar patterns without manual feature engineering.

## 🧠 Technologies

- Python, PyTorch, NumPy
- Matplotlib / Seaborn
- Jupyter Notebooks

## 🧪 Team

- Artem Chuprov  
- Zhemchueva Darya  
- Pavlov Nikita  
- Sausar Karaf

## Network model

![Network model](img/Clustering%20Time-Series%20Data%20via%20Neural%20Networks.png)

## Results

After training the neural network was able to clusterize the stock behavior according to the stock volatility.


| Cluster | Delta   | Step Volatility | Total Volatility |
|---------|---------|------------------|------------------|
| 0       | 0.005   | 0.0076           | 0.0118           |
| 1       | -0.0004 | 0.0087           | 0.0142           |
| 2       | 0.0263  | 0.0117           | 0.0216           |


## 🏁 Getting Started

```bash
git clone https://github.com/yourusername/deep-metric-ts-clustering.git
cd deep-metric-ts-clustering
pip install -r requirements.txt