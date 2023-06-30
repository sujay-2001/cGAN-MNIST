# cGAN-MNIST
## Abstract:
This blog post provides an overview of the knowledge and skills I acquired during my personal project focused on the MNIST dataset. Specifically, I explored the study of Generative Adversarial Networks (GANs), model parameters, and hyperparameters in this context. I will highlight the implementation of GANs on the MNIST dataset and my exploration of hyperparameter tuning for improved results.

Throughout this project, I delved into the fundamentals of GANs, a powerful deep learning model used for generating synthetic data. I gained an understanding of their significance in various applications, particularly in image generation and data augmentation tailored to the MNIST dataset.

I will discuss the practical implementation of GANs on the MNIST dataset, following a step-by-step process to generate realistic images. By experimenting with different model architectures and training techniques, I improved the quality of the generated images while considering the unique characteristics of the MNIST dataset.

Furthermore, I explored the importance of model parameters and hyperparameters in the training process. I learned how these factors impact the performance and quality of the generated images specifically for the MNIST dataset. Through meticulous hyperparameter tuning, I made adjustments to enhance the performance of the GAN models and generate more compelling and realistic images.

As future work, I propose leveraging the skills I acquired to further enhance the generation of synthetic images on the MNIST dataset. By fine-tuning the model parameters and hyperparameters, I aim to push the boundaries of image generation and explore novel applications, such as data augmentation for MNIST-based tasks.

This blog post showcases my personal project journey, highlighting the practical implementation of GANs on the MNIST dataset. It discusses the impact of model parameters and hyperparameters specific to MNIST, while outlining the potential future applications of these skills in image generation and data augmentation for this dataset.

## Implementation of cGAN on MNIST Dataset:

> **Description of the MNIST Dataset:** The MNIST dataset is a widely used benchmark dataset in the field of machine learning. It stands for “Modified National Institute of Standards and Technology” and consists of a collection of handwritten digits from 0 to 9. The dataset was created using samples of handwritten digits from various sources, including high school students and Census Bureau employees. The MNIST dataset contains a training set of 60,000 images and a test set of 10,000 images. Each image is grayscale and has a resolution of 28x28 pixels, resulting in a total of 784 features. The images are labeled with the corresponding digit they represent, ranging from 0 to 9.

**Dataset Preparation:** The first step in the implementation process was to prepare the MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. Each image is a grayscale image of size 28x28 pixels. I used the Tensorflow framework to load and preprocess the dataset.
**Building the Generator Network:** The generator network is responsible for generating fake images that resemble the real images from the dataset. I implemented a deep convolutional neural network (CNN) as the generator network. The network takes a random noise vector along with a label (condition) as input and generates an image of size 28x28 pixels. The final architecture is as shown below.
![Demo Link](https://github.com/sujay-2001/cGAN-MNIST/blob/main/mnist_gen.png)

**Building the Discriminator Network:** The discriminator network is responsible for distinguishing between real and fake images. It is also implemented as a deep CNN. The discriminator network takes an image along with a label (condition) as input and outputs a probability indicating whether the image is real or fake.
![Demo Link](https://github.com/sujay-2001/cGAN-MNIST/blob/main/mnist_dis.png))
  
**Training the GAN:** The GAN training process involves training the generator and discriminator networks in an adversarial manner. The training process I employed involves alternating between training the discriminator on real and fake images and training the generator to improve its ability to generate realistic images. The loss function used is Binary Cross Entropy and optimiser is Adam. Overall the parameters of Generator and Discriminator are updated based on the following equation:

**Evaluation and Tuning the Hyper-parameters of the model:** After training the GAN on the MNIST dataset, I evaluated the performance of the generator network by generating a set of fake images for each class and comparing them. I also observed how well the model is able to learn the condition, and how it performs with respect to controlled generation. I also calculated metrics such as accuracy and loss to measure the performance of the discriminator and generator networks. I used the above metrics to tune the hyperparameters of the model. This is one of the most challenging task I had to work on.

Hyperparameters:

> Learning Rate (=0.001): lr = 0.01 led to unstable training (generator unable to learn with discriminator), while lr = 0.0001 led to slower training, After careful consideration, lr = 0.001 was chosen.

> Batch Size (=128): Choosing batch size of 128 showed stable learning, while any sizes greater or lesser led to unstable training. Another important observation is that batch size has significant correlation with learning rate with respect to performance.

> Batch Normalisation Layers: Batch Normalisation layers were added after every layer’s output before feeding it to the next layer, without which the model wasn’t able to learn as shown.

> Choice of Activation functions: Relu in Generator network and LeakyRelu with α = 0.2 in Discriminator network gave best results as shown.

> Dropout =0.4: Addition of dropouts in final flattened layer of discriminator network improved results.
> No of Epochs =50: Results are good after 50 epochs with 128 as batch size.

**Results:** The quality of the generated images is satisfactory and distinguishable. The final accuracy of the model is evaluated by computing the mean square error (MSE) of the normalized pixel values of generated image and real images for each class. The evaluated accuracy is 89%..

## Extension ideas:
Effect of changing latent space.

Effect of introducing skip connections.

Effect of adding L2/SSIM loss.
