import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.cluster.vq import kmeans2, whiten
from sklearn.decomposition import PCA
import image_loader as loader
import pickle
import matplotlib.pyplot as plt
MODEL_LOG_FOLDER = "drive/Shareddrives/Perovskites_DIRECT/models"
dataset = loader.PLDataLoader()

rc = {'figure.figsize':(10,5),
      'axes.facecolor':'white',
      'axes.grid' : True,
      'grid.color': '.8',
      'font.family':'DejaVu Sans',
      'font.size' : 15}
plt.rcParams.update(rc)

class autoencoder:
    def __init__(self, data=dataset, h5_name='Autoencoder.h5'):
        self.h5_name = h5_name
        self.data = data

    def extract_autoencoder(self,
                            optimizer,
                            epochs=100,
                            batch_size=150,
                            file_path=MODEL_LOG_FOLDER):
        """
        Takes an input of multiple numpy arrays representing images
        and returns a fitted autoencoder.

        Parameters
        -------------------
        optimizer:
            What optimizer fuction the autoencodre uses to train the model
        data: Three dimensional numpy array
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
            which can be used to construct encoded versions of given input
            images.

        """
        split = self.data.train_test_split(
                                 test_size=0.2,
                                 random_state=42,
                                 return_dfs=True)

        train_X = split[0]/split[0].max()
        valid_X = split[1]/split[1].max()

        input_img = tf.keras.Input(shape=(32, 32, 1))
        stride = (3, 3)  # Change stride

        x = layers.Conv2D(32, stride,
                          activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, stride, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, stride, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        encoded = layers.Conv2D(4, stride,
                                activation='relu', padding='same')(x)

        x = layers.Conv2D(4, stride,
                          activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(8, stride, activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, stride, activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, stride, activation='relu', padding='same')(x)
        decoded = layers.Conv2D(1, (2, 2),
                                activation='sigmoid', padding='same')(x)

        decoder = tf.keras.Model(input_img, decoded)
        decoder.compile(optimizer=optimizer, loss='binary_crossentropy')
        decoder.summary()
        history = decoder.fit(train_X,
                        train_X,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(valid_X, valid_X))

        encoder = tf.keras.Model(input_img, encoded)
        decoder.save_weights(file_path +
                                 '/autoencoder_model/' +
                                 self.h5_name)
        encoder.save_weights(file_path + '/encoder_model/' + self.h5_name)
        
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss as a function of Epochs')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss [Binary Crossentropy]')
        plt.savefig(file_path + '/encoder_model/loss_graph.png')
        return decoder, encoder

    def core_autoencoder_fxn(self,
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
        pickle_output = self.data.sample(frac=1.0, return_dfs=True)
        full_dataset = pickle_output[0]/pickle_output[0].max()
        full_dataset_labels = list(pickle_output[3].index)
        decoder, encoder = self.extract_autoencoder(optimizer,
                                                    epochs,
                                                    batch_size)
        encoded_imgs = encoder.predict(full_dataset)
        output_array = []
        for enc_img in encoded_imgs:
            output_array.append(enc_img.flatten())
        return np.array(output_array), full_dataset_labels

    def Kmeans_clustering(self,
                          encoded_array,
                          centroids=10,
                          iter=20):
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
            A list of the index for the image and the cluster that it
            belongs to
        list_of_centroids:
            The list of centroids in ascending order corresponding to
            the order of labels fed into the function
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

    def PCA_dimension_reduction(self,
                                PCA_input,
                                PCA_dims):
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

    def autoencoder_to_classification(self,
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

        autoencoder_output = self.core_autoencoder_fxn(epochs,
                                                       batch_size,
                                                       optimizer)
        encoded_array = autoencoder_output[0]
        train_indecies = autoencoder_output[1]

        if run_PCA is True:
            encoded_array_PCA = self.PCA_dimension_reduction(encoded_array,
                                                             PCA_dims)
            cluster_assignment = self.Kmeans_clustering(encoded_array_PCA,
                                                        centroids,
                                                        iter)[0]
            clusters = self.Kmeans_clustering(encoded_array_PCA,
                                              centroids,
                                              iter)[1]
            autoencoder_output_PCA = (train_indecies,
                                      encoded_array,
                                      cluster_assignment,
                                      clusters,
                                      encoded_array_PCA)
            with open('drive/Shareddrives/Perovskites_DIRECT' +
                      '/autoencoder_output_PCA.pickle', 'wb') as f:
                pickle.dump(autoencoder_output_PCA, f)
            return autoencoder_output_PCA

        cluster_assignment = self.Kmeans_clustering(encoded_array,
                                                    centroids,
                                                    iter)[0]
        clusters = self.Kmeans_clustering(encoded_array, centroids, iter)[1]
        autoencoder_output_noPCA = (train_indecies,
                                    encoded_array,
                                    cluster_assignment,
                                    clusters)
        with open('drive/Shareddrives/Perovskites_DIRECT' +
                  '/autoencoder_output_noPCA.pickle', 'wb') as f:
            pickle.dump(autoencoder_output_noPCA, f)
        return autoencoder_output_noPCA
    
    def build_autoencoder(self, epochs=100,
                          batch_size=150,
                          optimizer='adam'):
        '''
        This builds a blank encoded and decoded model that can
        subsequently be used to load a model with keras (that has
        already been trained.)
        
        Parameters
        -------------------
        epochs: int
            Number of epochs that you desire your function to run
        batch_size: int
            Number of images per batch ran through your epoch
        optimizer:
            What optimizer fuction the autoencodre uses to train the model

        Returns
        -------------------
        decoder:
            Model class -- blank decoded model with correct shape

        encoder:
            Model class -- blank encoded model with correct shape

        '''

        
        input_img = tf.keras.Input(shape=(32, 32, 1))
        stride = (3,3) # Change stride

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

        encoder = tf.keras.Model(input_img, encoded)
        encoder.compile(optimizer=optimizer, loss='binary_crossentropy')
        return autoencoder, encoder
