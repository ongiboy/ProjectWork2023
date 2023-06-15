import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def PCA_embeddings(embeddings): # input numpy arrays, shape (x,2)
    #n_components is number of PC's
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    embeddings_pca = pca.transform(embeddings)
    return embeddings_pca

if __name__ == "__main__":
    modes = ["pretraining", "finetuning"]
    mode = modes[1]
    array_number = "9868"
    plot = ""

    #labels only needed if mode == finetune
    z_t = np.load("code/PCA_embeddings/{}/{}_{}z_t.npy".format(mode,array_number,plot))
    #z_t_aug = np.load(f"code/PCA_embeddings/{mode}/z_t_aug.npy")
    z_f = np.load("code/PCA_embeddings/{}/{}_{}z_f.npy".format(mode,array_number,plot))
    #z_f_aug = np.load(f"code/PCA_embeddings/{mode}/z_f_aug.npy")

    embeddings = np.vstack((z_t, z_f))  # z_t_aug z_f_aug
    embeddings_pca = PCA_embeddings(embeddings)

    plt.figure(figsize=(8, 6))
    if mode == "finetuning":
        labels = list(np.load("code/PCA_embeddings/{}/{}_{}labels.npy".format(mode,array_number,plot))) * 2
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            # class coloring
            plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], label=label)

    elif mode == "pretraining":
        plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')
    plt.legend()
    plt.show()