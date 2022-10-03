# Project : Photo Eraser

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
  - Use Phanteon, St. Peters Square, Colosseum

### Test Data

- Kaggle Image Matching Challenge 2022 Dataset
  - link : https://www.kaggle.com/competitions/image-matching-challenge-2022/data
  - Use Phanteon, St. Peters Square, Colosseum
- My Pictures

## Data Augmentation

- Create 12-Black Masks(64x64) for each train image.
  ![img]("./figures/Custom_DA_sample.png")

## Result

- Restore Image.
- But, Low Resolution Image

## More

- If Better Computing Env...
  - Deeper Network.
  - More Batch Size.
  - Bigger Image Size.
