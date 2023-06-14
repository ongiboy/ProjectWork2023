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
    modes = ["pretrain", "finetune"]
    mode = modes[0]

    #labels only needed if mode == finetune
    labels = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0,
 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1,
 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0,]*4
    embeddings_pca=[[1.3401886e+00,-1.9016058e+00],
[1.1631744e+00,-1.7177079e+00],
[6.8045956e-01,-2.2093325e+00],
[1.0609857e+00,-1.4952910e+00],
[1.4177675e+00,-1.7230065e+00],
[8.6498439e-01,-1.4751688e+00],
[7.6046932e-01,-1.1734183e+00],
[2.6360185e+00,-4.6007359e-01],
[1.6983016e+00,-4.8314780e-01],
[4.7654386e+00,2.0590169e+00],
[1.9585087e+00,-1.9930266e-01],
[8.8936108e-01,-1.2592455e+00],
[1.4125053e+00,-1.5160439e+00],
[3.1209488e+00,8.9662009e-01],
[7.2795987e-01,-1.2073715e+00],
[1.1312320e+00,-7.0648563e-01],
[1.2487020e+00,-1.2031924e+00],
[4.8733978e+00,1.9857922e+00],
[1.8238280e+00,-5.7884598e-01],
[1.8021210e+00,-9.0797842e-01],
[1.0044978e+00,-1.5552846e+00],
[1.3097644e+00,-1.5562066e+00],
[1.3469745e+00,-1.5051085e+00],
[1.7938687e+00,2.0741904e-01],
[2.6757877e+00,1.1787450e+00],
[8.6939216e-01,-1.3844463e+00],
[3.1656149e-01,-1.6079297e+00],
[1.7933595e+00,-7.7411562e-01],
[4.8002973e+00,3.4188852e+00],
[3.6071630e+00,2.2389562e+00],
[6.8147197e+00,4.1534948e+00],
[4.7205753e+00,2.8509960e+00],
[1.1076370e+00,-1.2174057e+00],
[1.9706162e+00,-1.1985958e+00],
[1.2098203e+00,-1.5601662e+00],
[1.2574764e+00,-1.2847158e+00],
[1.0848137e+00,-1.5046067e+00],
[1.3065115e+00,-1.8033938e+00],
[9.8726815e-01,-1.2188041e+00],
[9.4984257e-01,-1.0602931e+00],
[1.0341194e+00,-1.2594774e+00],
[2.0087502e+00,2.4915550e-02],
[7.1362537e-01,-1.5316224e+00],
[1.3712806e+00,-8.1355488e-01],
[1.0893161e+00,-1.0896356e+00],
[7.0585769e-01,-9.6678352e-01],
[9.3131369e-01,-1.1563066e+00],
[1.4403105e+00,-1.6486753e+00],
[1.1618766e+00,-5.9889120e-01],
[1.2783425e+00,-7.7736533e-01],
[9.1621530e-01,-1.9073381e+00],
[1.1206869e+00,-1.3599031e+00],
[1.5012498e+00,-1.7733119e+00],
[1.3727829e+00,-6.3344061e-01],
[2.7308569e+00,7.4640465e-01],
[7.0932932e+00,4.2820706e+00],
[1.2874458e+00,-1.3591415e+00],
[1.0676416e+00,-1.4811556e+00],
[2.5236537e+00,-2.6522607e-01],
[2.1182296e+00,6.0692585e-01],
[7.4035460e-01,-1.2373289e+00],
[1.2412671e+00,-1.5185249e+00],
[1.2000479e+00,-1.3102791e+00],
[1.1284916e+00,-1.0483756e+00],
[1.0454868e+00,-8.4808314e-01],
[1.1106179e+00,-1.3584621e+00],
[5.7176477e-01,-1.6516790e+00],
[2.2454808e+00,-9.4841145e-02],
[1.2970713e+00,-7.9809111e-01],
[3.8549061e+00,1.9903336e+00],
[2.1861227e+00,2.0455880e-01],
[1.3140701e+00,-1.6134391e+00],
[7.8987831e-01,-1.5493920e+00],
[2.6683002e+00,8.3995241e-01],
[8.2518762e-01,-1.5108092e+00],
[8.9753109e-01,-1.3180809e+00],
[1.1261963e+00,-1.3062592e+00],
[4.9752254e+00,2.3826509e+00],
[1.7119492e+00,-5.0800455e-01],
[1.3510793e+00,-1.1862993e+00],
[1.1168227e+00,-1.3808428e+00],
[1.1859593e+00,-1.7122598e+00],
[9.3863028e-01,-1.8275466e+00],
[2.5194883e+00,-7.1986282e-01],
[2.8585048e+00,1.5697883e+00],
[1.1039472e+00,-1.1271707e+00],
[9.4323421e-01,-1.9009151e+00],
[2.0883770e+00,-8.9703321e-01],
[5.9408484e+00,4.6508183e+00],
[4.7537012e+00,2.5822549e+00],
[6.0376563e+00,4.5254364e+00],
[4.7019067e+00,2.3611979e+00],
[1.1534827e+00,-1.5234963e+00],
[1.7331696e+00,-3.7075680e-01],
[1.3331635e+00,-1.6776079e+00],
[1.0968804e+00,-1.3511403e+00],
[1.0462337e+00,-1.0452452e+00],
[1.2138673e+00,-1.3496009e+00],
[1.0969425e+00,-1.1890095e+00],
[9.5979661e-01,-2.4540555e+00],
[9.1621423e-01,-1.2278826e+00],
[1.7866179e+00,-1.0751935e+00],
[1.4562107e+00,-1.0265621e+00],
[1.5192962e+00,-1.8832436e+00],
[1.3038733e+00,-1.5451548e+00],
[1.4457151e+00,-1.2761208e+00],
[1.2187796e+00,-1.2826117e+00],
[9.2891085e-01,-1.4568595e+00],
[1.7079536e+00,-1.1287138e+00],
[1.6037053e+00,-5.9771347e-01],
[9.6537852e-01,-2.0021670e+00],
[1.0540949e+00,-1.2989244e+00],
[8.6220062e-01,-1.3743441e+00],
[1.5728369e+00,-7.2513551e-01],
[2.3004880e+00,7.7401888e-01],
[7.8905039e+00,4.4368448e+00],
[1.8951797e+00,-1.1208220e+00],
[7.4822837e-01,-1.5956411e+00],
[1.7664306e+00,-8.7211257e-01],
[1.9720976e+00,5.9888214e-01],
[-2.5897141e+00,-1.3763086e-01],
[-2.6597555e+00,4.4351617e-01],
[-1.8621231e+00,3.4730369e-01],
[-1.9847599e+00,7.6214296e-01],
[-2.1092579e+00,1.8298410e-01],
[-2.4806924e+00,8.3493739e-01],
[-1.8794345e+00,3.3411449e-01],
[-1.7692076e+00,7.3577218e-02],
[-1.6324662e+00,-5.7468578e-02],
[-2.7047291e+00,2.1966541e-01],
[-2.4055760e+00,2.7336138e-01],
[-2.5733082e+00,7.8681475e-01],
[-1.6748160e+00,2.4865519e-01],
[-2.2202914e+00,1.1094099e+00],
[-1.8482388e+00,2.9515076e-01],
[-1.0551915e+00,3.4841183e-01],
[-2.1064808e+00,7.7530605e-01],
[-1.2938418e+00,7.6430574e-02],
[-1.6071954e+00,9.6760023e-01],
[-1.7630352e+00,9.3425918e-01],
[-1.3862218e+00,7.7029890e-01],
[-2.7363429e+00,9.3565887e-01],
[-1.8778681e+00,5.0641376e-01],
[-1.2487073e+00,-3.0280875e-02],
[-1.7173287e+00,7.0962906e-01],
[-1.8260072e+00,1.1299129e+00],
[-1.3117952e+00,6.2837738e-01],
[-1.5377414e+00,2.4637774e-01],
[-2.1858034e+00,1.3163825e+00],
[-1.8492255e+00,6.4746022e-01],
[-2.9610379e+00,1.0728889e+00],
[-2.8200154e+00,7.1697158e-01],
[-1.8133056e+00,1.5918931e+00],
[-1.7733604e+00,2.4976952e-01],
[-2.4099157e+00,1.4698350e+00],
[-1.5872003e+00,5.5101883e-01],
[-2.0322015e+00,1.8583426e-01],
[-1.5855837e+00,1.0165315e+00],
[-1.4739572e+00,2.8504247e-02],
[-1.6597308e+00,7.9311180e-01],
[-1.8280514e+00,1.7183964e-01],
[-1.0995442e+00,-1.7930457e-01],
[-1.1618809e+00,4.0216580e-01],
[-1.5186734e+00,-2.3460828e-01],
[-1.5224622e+00,1.0701845e-01],
[-1.8687716e+00,5.7972556e-01],
[-2.1215916e+00,1.0987673e+00],
[-1.2792683e+00,1.0031004e+00],
[-1.8235809e+00,1.3609242e+00],
[-1.9746829e+00,-3.2113737e-01],
[-2.1171110e+00,9.8570061e-01],
[-1.2414743e+00,3.4749943e-01],
[-1.8400183e+00,1.7319970e+00],
[-1.6174657e+00,7.1937698e-01],
[-1.1876093e+00,5.4182583e-01],
[-1.2232878e+00,-3.9137481e-03],
[-1.7900170e+00,1.3437216e-01],
[-1.1713976e+00,9.7079599e-01],
[-2.1146584e+00,5.3757444e-02],
[-2.0724180e+00,4.1636553e-01],
[-2.2022550e+00,1.1608969e+00],
[-1.3360720e+00,5.0797302e-01],
[-1.7379829e+00,3.1682637e-01],
[-1.6244220e+00,9.9641234e-01],
[-1.6888256e+00,3.2838443e-01],
[-2.0687714e+00,6.2133467e-01],
[-1.9925330e+00,7.3434746e-01],
[-2.2655308e+00,1.3450174e+00],
[-1.5959595e+00,8.9579469e-01],
[-1.9009105e+00,8.6601865e-01],
[-1.8058878e+00,5.2842200e-02],
[-1.6804705e+00,-2.9415101e-01],
[-3.0507152e+00,1.1214818e+00],
[-2.4564996e+00,4.1814071e-01],
[-2.0134542e+00,2.3004718e-01],
[-1.9534537e+00,2.1002977e-01],
[-2.3786390e+00,5.3511113e-01],
[-2.1220732e+00,6.3119775e-01],
[-1.5354139e+00,1.0986922e+00],
[-1.5858188e+00,-3.4767920e-01],
[-2.0601678e+00,5.7816368e-01],
[-2.2292736e+00,4.6712494e-01],
[-1.6111109e+00,4.8549944e-03],
[-1.5137355e+00,8.5578763e-01],
[-1.9728622e+00,1.0881488e-01],
[-1.4780998e+00,2.3062915e-01],
[-1.3731999e+00,6.3012415e-01],
[-2.1501021e+00,7.9981941e-01],
[-1.5971107e+00,4.8928842e-01],
[-1.2147528e+00,6.8966061e-01],
[-2.1294734e+00,9.7556990e-01],
[-2.3107677e+00,7.7951342e-01],
[-2.6334903e+00,-2.1963689e-01],
[-1.9058604e+00,6.1732084e-01],
[-1.8182647e+00,-4.1338247e-01],
[-2.4597507e+00,4.2289278e-01],
[-1.2088034e+00,4.2227438e-01],
[-1.3073224e+00,2.9751849e-01],
[-1.8321545e+00,9.5197034e-01],
[-1.4601300e+00,5.8054382e-01],
[-2.2461262e+00,9.8137283e-01],
[-1.2961379e+00,1.0104152e+00],
[-1.0462986e+00,5.0840372e-01],
[-1.0851313e+00,8.1476545e-01],
[-1.2056841e+00,6.3355613e-01],
[-9.9870247e-01,-2.4776697e-01],
[-1.4882555e+00,5.7116419e-01],
[-1.6868184e+00,3.3450097e-01],
[-2.1661837e+00,2.0643426e-01],
[-2.1689651e+00,9.7508585e-01],
[-2.3188279e+00,1.5181761e+00],
[-1.7127734e+00,4.8849192e-01],
[-2.0471854e+00,7.2104025e-01],
[-1.9315234e+00,1.4219797e+00],
[-2.3268557e+00,1.0172347e+00],
[-1.3113692e+00,1.8709296e-02],
[-1.9786242e+00,6.0337782e-01],
[-2.1276622e+00,4.2019656e-01],
[-2.0332584e+00,8.7240678e-01],
[-1.7625580e+00,5.6831235e-01]]
    

    #embeddings_pca = PCA_embeddings(embeddings)

    plt.figure(figsize=(8, 6))
    if mode == "finetune":
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            # class coloring
            plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], label=label)

    elif mode == "pretrain":
        plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')
    plt.legend()
    plt.show()