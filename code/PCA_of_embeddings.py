import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mode = "pre_trains"  # finetune or pretrain

#labels only needed if mode == finetune
labels = [2, 1, 2, 2, 1]
embeddings = [[-0.0881,  0.3651,  0.4371,  0.2529, -0.0125,  0.1860],
        [-0.2561,  0.4303,  0.4384,  0.2136, -0.0061,  0.2175],
        [-0.2152, -0.3515, -0.0453, -0.2395, -0.4993,  0.1085],
        [-0.1353,  0.3933,  0.4766,  0.2137,  0.0296,  0.2342],
        [-0.3589,  0.4560,  0.3395,  0.1862, -0.0783,  0.2093]]


#n_components is number of PC's
pca = PCA(n_components=2)
pca.fit(embeddings)
embeddings_pca = pca.transform(embeddings)


plt.figure(figsize=(8, 6))
if mode == "finetune":
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], label=label)

elif mode == "pretrain":
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.legend()
plt.show()