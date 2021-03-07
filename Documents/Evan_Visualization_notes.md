To visualize layers of keras model:
  from IPython.display import SVG
  from keras.utils.vis_utils import model_to_dot

  SVG(model_to_dot(model).create(prog='dot', format='svg'))

model = VGG16:

Visualize convolutional filter:
  Useful to understand  regularities captured in the first conv layer of a network, such as edges and color patterns. Note that regularities for later layers become more complex and less interpretable.

  visualization.viz_conv_filters(model, ["block1_conv1", "block1_conv2"])

Visualize output of convolutional and pooling layers given input image:
  Shows how filters extract and decompose the information enclosed in the image. Each layer is formed by a number of 2D filters. The activation of each filter is shown as a separate image.
  Note that the target image is preprocessed by applying the same transformations used for the training images, thus ensuring accurate results. This is done by passing the VGG16 preprocessing function as argument f_preproc:

  visualization.viz_activations("./images/kitten.jpg",
                model,
                ["block1_conv1", "block2_pool"],
                img_size=(224, 224),
                f_preproc=preprocess_input,
                fig_size=(40,40))

Visualize nearest neighbors:
  target_images =["images/n02419796_11570_antelope.jpg", "images/doberman.png", "images/n02510455_7939_giant_panda.jpg"]
  index_images = list(set(glob.glob("images/*")) - set(target_images))

  visualization.viz_nearest_neighbors(target_images,
                                    index_images,
                                    model,
                                    "fc1",
                                    k=5,
                                    img_size=(224, 224, 3),
                                    f_preproc=preprocess_input,
                                    fig_size=(20,60))


Visualization of saliency maps:
  visualization.viz_saliency_map("./images/n02084071_77_dog.jpg", model, 281, img_size=(224, 224, 3),
                             f_preproc=preprocess_input, variant="vanilla", multiply=False, pair=True)

Calculating Statitics for parity plot:

  mean_abs_err = np.mean(np.abs(x-y))
  rmse = np.sqrt(np.mean((x-y)**2))
  rmse_std = rmse / np.std(y)
  z = np.polyfit(x,y, 1)
  y_hat = np.poly1d(z)(x)
