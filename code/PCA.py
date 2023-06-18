import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## Create PCA embeddings for the point embeddings of the training and test set and plotting them##

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
            z_f = np.load("code/PCA_embeddings/{}/{}_{}{}_z_f.npy".format(mode,array_number,plot,test))
            
            embeddings = np.hstack((z_t, z_f)) 
            embeddings_pca = PCA_embeddings(embeddings)

            if mode == "finetuning":
                labels = list(np.load("code/PCA_embeddings/{}/{}_{}{}_labels.npy".format(mode,array_number,plot,test)))
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    # class coloring
                    axs[p,t].scatter(embeddings_pca[mask, PC[0]], embeddings_pca[mask, PC[1]], label=label, s=15)

            axs[p,t].set_title("{} point embeddings {} finetuning".format(name_test[t], name_plot[p]))
    for ax in axs.flat:
        ax.set(xlabel='PC{}'.format(PC[0]+1), ylabel='PC{}'.format(PC[1]+1))
    plt.legend()
    plt.show()