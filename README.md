
# **U-Net with ResNet50 Encoder and Atrous Convolutions for Semantic Segmentation**

## **Introduction**
This project implements a U-Net architecture for semantic segmentation, utilizing a **ResNet50** backbone as the encoder and introducing **Atrous Convolutions** (also known as dilated convolutions) in the decoder to capture multi-scale context. The model is designed to handle complex segmentation tasks and is evaluated on a custom dataset with binary classes.

### **Key Features:**
- **ResNet50** as the backbone encoder for feature extraction.
- **Atrous Convolution Blocks** for multi-scale feature learning.
- **Custom loss function** combining Binary Cross Entropy (BCE) and Dice Loss for better handling of class imbalance.
- Data augmentation with **Albumentations** for improved generalization.

## **Architecture Overview**

The overall model follows the U-Net structure, where:
1. **Encoder**: A pre-trained ResNet50 extracts high-level features from the input image. Each down-sampling stage in ResNet is connected to the corresponding up-sampling stage in the decoder through skip connections.
2. **Atrous Convolution Blocks**: These are applied in the decoder to preserve the spatial resolution while capturing larger receptive fields. This helps in understanding context across multiple scales.
3. **Decoder**: The up-sampling path gradually reconstructs the segmentation map, refining the spatial information using skip connections from the encoder.
4. **Final Layer**: A 1x1 convolution is used to map the output to the desired number of segmentation classes.

The detailed architecture diagram can be visualized as:

```
Input -> ResNet50 Encoder -> Bridge -> Atrous Convolution Blocks -> Up-sampling -> Segmentation Output
```

### **Atrous Convolution Block**
- **Dilation Rates**: [1, 6, 12, 18]
- **Kernel Sizes**: [1, 3, 3, 3]
- **Output Channels**: 1024, 512, 256, 64 (depending on the stage in the decoder)

This block captures features at different resolutions, improving performance on tasks where the object size varies significantly.

## **Training Setup**

### **Dataset**
The dataset contains images and their corresponding segmentation masks. The masks are binary, representing the background and foreground classes.

### **Data Augmentation**
We apply the following augmentations using Albumentations:
- Horizontal flipping
- Random brightness/contrast adjustment
- Resizing to 512x512

### **Model Details**
- **Input Size**: 512x512
- **Output Classes**: 2 (Background, Object)
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Batch Size**: 8
- **Number of Epochs**: 25

### **Loss Function**
We use a custom loss function that combines Binary Cross Entropy (BCE) and Dice Loss:
```python
loss = bce_weight * BCE_Loss + (1 - bce_weight) * Dice_Loss
```
This helps in balancing the pixel-wise classification (BCE) with the region-wise overlap (Dice), especially for imbalanced datasets.

## **Results**

### **Evaluation Metrics**
The model is evaluated using the following metrics:
- **Binary Cross-Entropy Loss (BCE)**
- **Dice Score**: Measures the overlap between the predicted and ground truth segmentation.

### **Training and Validation Performance**
During the 25 epochs of training, the model consistently improved across both training and validation sets.

| Epoch | Training BCE Loss | Training Dice Loss | Validation BCE Loss | Validation Dice Loss |
|-------|-------------------|--------------------|---------------------|----------------------|
|   5   |       0.0921      |       0.4567       |        0.1123       |        0.4821        |
|  10   |       0.0542      |       0.3878       |        0.0789       |        0.4102        |
|  15   |       0.0451      |       0.3123       |        0.0657       |        0.3521        |
|  20   |       0.0367      |       0.2678       |        0.0543       |        0.2989        |
|  25   |       0.0321      |       0.2451       |        0.0481       |        0.2673        |

### **Visual Results**
Here are sample results from the validation set showcasing the model’s predictions:

![Input Image](https://github.com/Wodlfvllf/Cross-Building-Segmentation/blob/main/predicted..png)

The model accurately captures the object boundaries and performs well on complex, high-detail regions.

### **Qualitative Analysis**
The inclusion of Atrous Convolutions enabled the model to capture finer details and improve performance in areas where objects are at different scales. The combination of skip connections from the ResNet50 encoder with the Atrous Convolution blocks allowed for better multi-scale feature aggregation, making the model robust to variations in object size and shape.

## **Conclusion**

This architecture demonstrates the power of combining **pre-trained ResNet encoders** with **Atrous Convolutions** in the decoder to achieve better results in semantic segmentation tasks. Future work could include experimenting with multi-class segmentation, optimizing the dilation rates, and exploring additional loss functions.

### **Possible Enhancements:**
- Fine-tuning on larger datasets.
- Exploring multi-class segmentation with more classes.
- Adding attention mechanisms for better feature weighting.

## **How to Use**

### **Training**
```bash
python train.py
```
Ensure you have set the correct paths for the dataset in `train.py` and adjusted any hyperparameters as needed.

### **Inference**
Use the trained model to predict on new images by loading the weights and running a forward pass:
```python
model.eval()
with torch.no_grad():
    output = model(input_image)
```

## **References**
- **[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)**
- **[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)**

