import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def PCA_embeddings(embeddings): # input numpy arrays, shape (x,2)
    #n_components is number of PC's
    pca = PCA(n_components=5)
    pca.fit(embeddings)
    embeddings_pca = pca.transform(embeddings)
    return embeddings_pca

if __name__ == "__main__":
    modes = ["pretraining", "finetuning"]
    mode = modes[1]
    array_number = "6127" # Insert embed ID here
    PC = [0,1]
    
    name_test = ["training", "test"]
    name_plot = ["before", "after"]
    
    fig, axs = plt.subplots(2,2)
    plt.subplots_adjust(hspace=0.4)
    for t,test in enumerate(["", "_test"]):
        for p,plot in enumerate(["bef", "aft"]):
            #labels only needed if mode == finetune
            z_t = np.load("code/PCA_embeddings/{}/{}_{}{}_z_t.npy".format(mode,array_number,plot,test))
            #z_t_aug = np.load(f"code/PCA_embeddings/{mode}/z_t_aug.npy")
            z_f = np.load("code/PCA_embeddings/{}/{}_{}{}_z_f.npy".format(mode,array_number,plot,test))
            #z_f_aug = np.load(f"code/PCA_embeddings/{mode}/z_f_aug.npy")
            
            embeddings = np.hstack((z_t, z_f))  # z_t_aug z_f_aug
            embeddings_pca = PCA_embeddings(embeddings)

            if mode == "finetuning":
                labels = list(np.load("code/PCA_embeddings/{}/{}_{}{}_labels.npy".format(mode,array_number,plot,test)))
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    # class coloring
                    axs[p,t].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], label=label, s=15)

            # elif mode == "pretraining":
            #     plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])

            axs[p,t].set_title("{} point embeddings {} finetuning".format(name_test[t], name_plot[p]))
    for ax in axs.flat:
        ax.set(xlabel='PC{}'.format(PC[0]), ylabel='PC{}'.format(PC[1]))
    plt.legend()
    plt.show()