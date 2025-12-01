# Vision Transformer (ViT) — From Scratch

This project implements the **Vision Transformer (ViT)** architecture **from scratch**, based on the seminal paper:

> **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**  
> *Dosovitskiy et al., Google Brain, 2021*

While our implementation closely follows the original paper, it is important to note that **Vision Transformers are extremely data-hungry**. Training ViT models from scratch on small datasets like **CIFAR-10** typically leads to underfitting and limited accuracy.  

To overcome this limitation, we also **fine-tuned a pre-trained ViT model** (trained on a large-scale dataset) for the CIFAR-10 classification task. This approach significantly improved accuracy and convergence speed.

---

##  Architecture Overview

<p align="center">
  <img src="./Images/vit_figure.png" alt="Vision Transformer Architecture" width="700" height="700">
</p>

The architecture follows the original ViT-B/16 configuration with:
- Patch embedding (16×16)
- Positional encoding
- Multi-head self-attention layers
- MLP blocks
- Classification token ([CLS])
- Layer normalization and final dense classifier

---

##  Pretrained Model and Fine-Tuning

We used the **pre-trained ViT model** from Kaggle’s TensorFlow Hub:

🔗 [ViT-B16 Classification Model on Kaggle](https://www.kaggle.com/models/spsayakpaul/vision-transformer/TensorFlow2/vit-b16-classification/1)

This model was fine-tuned on CIFAR-10 for improved performance.  
The fine-tuning process used transfer learning techniques while keeping the transformer backbone frozen initially and unfreezing in later stages.

---

##  Example Usage

You can easily load and use the pre-trained model via TensorFlow Hub:

```python
import tensorflow_hub as hub
import tensorflow as tf

# Load the pretrained ViT model
model = tf.keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/spsayakpaul/vision-transformer/TensorFlow2/vit-b8-classification/1")
])

# Example prediction
predictions = model.predict(images)
