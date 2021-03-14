import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.cluster.vq import kmeans2, whiten
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def extract_autoencoder(optimizer,
                        data,
                        epochs=100,
                        batch_size=150,
                        file_path='/models'):
    """
    Takes an input of multiple numpy arrays representing images
    and returns a fitted autoencoder.

    Parameters
    -------------------
    optimizer:
        What optimizer fuction the autoencodre uses to train the model
    X_train: Three dimensional numpy array
        Array containing all 32x32 images that are to be fed into the model
    epochs: int
        Number of epochs that you desire your function to run
    batch_size: int
        Number of images per batch ran through your epoch

    Returns
    -------------------
    decoder:
        Model class -- decode model trained on the input images,
        which can be used to reconstruct images from a given input

    encoder:
        Model class -- encode model trained on input images,
        which can be used to construct encoded versions of given input images.

    """
    split = train_test_split(data,
                             data,
                             test_size=0.2,
                             random_state=42)
    train_X = split[0]
    valid_X = split[1]
    train_ground = split[2]
    valid_ground = split[3]

    input_img = tf.keras.Input(shape=(32, 32, 1))
    stride = (3, 3)  # Change stride

    x = layers.Conv2D(32, stride, activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, stride, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, stride, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(4, stride, activation='relu', padding='same')(x)

    x = layers.Conv2D(4, stride, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, stride, activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, stride, activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, stride, activation='relu', padding='same')(x)
    decoded = layers.Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    autoencoder.summary()
    autoencoder.fit(train_X,
                    train_ground,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(valid_X, valid_ground))

    encoder = tf.keras.Model(input_img, encoded)
    autoencoder.save(file_path + '/autoencoder_model')
    encoder.save(file_path + '/encoder_model')

    return autoencoder, encoder


def core_autoencoder_fxn(data,
                         epochs=100,
                         batch_size=150,
                         optimizer='adam'):
    """
    Given a data set this function will build an autoencoded
    CNN and return the encoded layer.

    Parameters
    -------------------
    data: Three dimensional numpy array
        Array containing all 32x32 images that are to be fed into the model
    epochs: int
        Number of epochs that you desire your function to run
    batch_size: int
        Number of images per batch ran through your epoch
    optimizer:
        What optimizer fuction the autoencodre uses to train the model

    Returns
    -------------------
    encoded_layer
        A 64x1 array with the values from the autoencoder
    """

    X_train = data/data.max()
    decoder, encoder = extract_autoencoder(optimizer,
                                           X_train,
                                           epochs,
                                           batch_size)
    encoded_imgs = encoder.predict(X_train)
    output_array = []
    for enc_img in encoded_imgs:
        output_array.append(enc_img.flatten())
    return np.array(output_array)


def encoded_Kmeans_clustering(encoded_array, centroids=10, iter=20):
    """
    Parameters
    -------------------
    encoded_array: two-dimensional dimensional numpy array
        Array where the rows are individual data points and the
        columns are the values from the encoded matrix
    centroids:
        Number of groups that the function will break your data into
    iter:
        Number of iuterations that the k-means clustering algorithm
        will carry out

    Returns
    -------------------
    list_of_classifications:
        A list of the index for the image and the cluster that it belongs to
    list_of_centroids:
        The list of centroids in ascending order
    """
    input_array = whiten(encoded_array)
    cluster_points, index_list = kmeans2(input_array,
                                         centroids,
                                         iter=iter,
                                         minit='points')
    pic_index_to_clusterID = []
    for index, val in enumerate(index_list):
        pic_index_to_clusterID.append((index, val))
    pic_index_to_clusterID = np.array(pic_index_to_clusterID)
    return pic_index_to_clusterID, cluster_points


def PCA_dimension_reduction(PCA_input, PCA_dims):
    '''
    Parameters
    -------------------
    PCA_input:
        Array where the rows are individual data points and the columns
        are the values from the encoded matrix
    PCA_dims:
        Number of demensions for the PCA function to reduce your
        encoded layer output to

    Returns
    -------------------
    output_matrix
        Returns a principal component reduction of the input matrix
        with dimensions = PCA_dims
    '''
    pca = PCA(n_components=PCA_dims)
    pca.fit(PCA_input)
    transformed = pca.transform(PCA_input)
    return transformed


def autoencoder_to_classification(data,
                                  epochs=100,
                                  batch_size=150,
                                  optimizer='adam',
                                  centroids=10,
                                  iter=20,
                                  run_PCA=True,
                                  PCA_dims=10):
    '''
    Parameters
    -------------------
    data: Three dimensional numpy array
        Array containing all 32x32 images that are to be fed into the model
    epochs: int
        Number of epochs that you desire your function to run
    batch_size: int
        Number of images per batch ran through your epoch
    optimizer:
        What optimizer fuction the autoencodre uses to train the model
    centroids:
        Number of groups that the function will break your data into
    iter:
        Number of iuterations that the k-means clustering algorithm will
        carry out
    run_PCA:
        Wheter or not we should run PCA on our components
    PCA_dims:
        Number of demensions for the PCA function to reduce your encoded
        layer output to

    Returns
    -------------------
    (list_of_classifications, )
    encoded_array:
        Returns an array of flattened encoded layers from the auto encoder
        of size len(data)x64
    list_of_classifications:
        Ordered list of the classifications for each datapoint
        of size len(data)x1
    list_of_centroids:
        A list of len(centroids) centroids, ordered numerically
        if "run_PCA" == False, of size len(data)x64
        if "run_PCA" == True, of size len(data)x"PCA_dims"
    encoded_array_PCA:
        Reduced array of dimension specified by "PCA_dims",
        will only be returned if "run_PCA" == True
    '''

    encoded_array = core_autoencoder_fxn(data, epochs, batch_size, optimizer)

    if run_PCA is True:
        encoded_array_PCA = PCA_dimension_reduction(encoded_array, PCA_dims)
        cluster_assignment = encoded_Kmeans_clustering(encoded_array_PCA,
                                                       centroids,
                                                       iter)[0]
        clusters = encoded_Kmeans_clustering(encoded_array_PCA,
                                             centroids,
                                             iter)[1]
        return (encoded_array, cluster_assignment, clusters, encoded_array_PCA)

    cluster_assignment = encoded_Kmeans_clustering(encoded_array,
                                                   centroids,
                                                   iter)[0]
    clusters = encoded_Kmeans_clustering(encoded_array, centroids, iter)[1]
    return (encoded_array, cluster_assignment, clusters)