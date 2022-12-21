# Project : Photo Eraser

- Restore Landmark Picture by Obstacle in front of Camera.

## My Laptop Env

- CPU : AMD Ryzen 9 5900HS
- RAM : 16GB
- GPU : RTX3070 Lapto GPU(8GB)

## Main Lib Environment

- Python : ver. 3.10.4
- Tensorflow : ver. 2.9.1
- opencv-python : ver. 4.6.0

## Data

### Train Data

- Kaggle Image Matching Challenge 2022 Dataset
  - link : https://www.kaggle.com/competitions/image-matching-challenge-2022/data
  - Use Train Colosseum Exterior (link:https://drive.google.com/drive/folders/1yLplmt9-ai9nmSz7sHUW4DPznj8d_dgg?usp=sharing)

<!-- ### Test Data

- Kaggle Image Matching Challenge 2022 Dataset
  - link : https://www.kaggle.com/competitions/image-matching-challenge-2022/data
  - Use Train Colosseum Exterior
- My Pictures -->

## Create Datasets

- Create 12-Black Masks(64x64) for each train image.
  <img src="./figures/Custom_DA_sample.png">

```
def create_blackbox(origin_size, idx):
    bbox_size = [64, 64]
    col = origin_size[1] // bbox_size[1]
    start_y = bbox_size[0] * (idx // col)
    end_y = bbox_size[0] + start_y
    start_x = bbox_size[1] * (idx % col)
    end_x = bbox_size[1] + start_x
    return start_y, end_y, start_x, end_x
```

## Model : Auto Encoder

- My Best Model link:https://drive.google.com/file/d/1k1QWk80_6o8rwzgAPfms9tdWrDvlRxlC/view?usp=sharing

```
class ConvAutoencoder(Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.autoenc_model, self.encoder, self.decoder = self.build_model()

    def build_model(self):
        input_img = layers.Input(shape=(192, 256, 3))
        x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(input_img)
        x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same")(x)
        encoded = layers.Conv2D(16, 3, strides=2, activation="relu", padding="same", name="encoded")(x)

        x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same")(encoded)
        x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
        decoded = layers.Conv2D(3, 3, activation="sigmoid", padding="same", name="decoded")(x)
        autoencoder = tf.keras.Model(input_img, decoded)
        autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError(), metrics=["accuracy"])
        encoder = tf.keras.Model(input_img, encoded)
        decoder = tf.keras.Model(encoded, decoded)
        return autoencoder, encoder, decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

## Result

- Restore Image with low resolution.
  <img src="./figures/result.png">
  <img src="./figures/result2.png">

## More

- If Better Computing Env...
  - Deeper Network.
  - More Batch Size.
  - Bigger Image Size.
- Use other GAN Models.
  - Cycle GAN.
  - SinGAN.
