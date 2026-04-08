## UNETR: Transformers for 3D Medical Image Segmentation

The **UNETR (UNEt TRansformers)** model is a highly effective architecture for 3D medical image segmentation that combines the global context-gathering power of Vision Transformers (ViTs) with the precise localization of a CNN-based U-Net decoder.

### 1. 3D Patch Embedding (The Input)
Instead of feeding the entire volume through convolutional layers, UNETR treats the 3D volume as a sequence of patches (similar to how NLP models process words).
- A 3D image volume (e.g., with size $H \times W \times D$) is divided into uniform, non-overlapping 3D patches (typically $16 \times 16 \times 16$ voxels).
- These 3D patches are flattened into 1D sequences and passed through a linear projection layer to map them into a fixed $K$-dimensional embedding space (e.g., $K=768$).
- A **1D learnable positional embedding** is added to each patch to ensure the model retains spatial awareness of where each patch belongs in the original 3D volume. Notably, it drops the classification `[class]` token used in standard ViTs since the goal is dense voxel-wise segmentation.

### 2. The Transformer Encoder (Global Context)
The embedded sequence is then passed through a pure Transformer encoder consisting of 12 blocks. Each block contains:
- **Multi-Head Self-Attention (MSA):** This allows the model to compute attention weights between *any two patches* in the entire 3D volume, regardless of their distance. This solves a major limitation of standard CNNs, which struggle to learn long-range dependencies due to their localized receptive fields.
- **Multilayer Perceptrons (MLP):** Standard feed-forward layers with GELU activations and layer normalization.
The resolution of the sequence remains completely fixed throughout all 12 transformer layers.

### 3. Skip Connections (The Bridge)
To capture multi-scale information (crucial for finding fine boundaries), the network extracts the hidden states from the Transformer at evenly spaced layers (specifically, layers 3, 6, 9, and 12).
- Because the Transformer processes 1D sequences, these extracted states are geometrically reshaped back into 3D spatial tensors (e.g., $\frac{H}{P} \times \frac{W}{P} \times \frac{D}{P} \times K$).
- These 3D tensors are then passed through consecutive $3 \times 3 \times 3$ convolutional layers to project them from the transformer's embedding space back into the image feature space.

### 4. The CNN Decoder (Precise Localization)
The decoder mirrors the expanding path of a standard U-Net. 
- It takes the deep, low-resolution features from the final Transformer layer and applies a **deconvolutional layer** to double the spatial resolution.
- This upsampled feature map is concatenated with the reshaped features extracted from the previous Transformer skip connection (e.g., layer 9).
- The combined features pass through $3 \times 3 \times 3$ CNN layers. This upsampling and concatenation process repeats until the feature map is restored to the original input resolution.
- Finally, a $1 \times 1 \times 1$ convolutional layer with a softmax activation predicts the semantic class for every single voxel.

### 5. Training and Loss
The paper trains the model using a combined loss function that works exceptionally well for class-imbalanced medical tasks:
- **Soft Dice Loss:** Evaluates the overlap between the prediction and the ground truth.
- **Cross-Entropy Loss:** Penalizes voxel-wise classification errors.
The model was optimized using the AdamW optimizer.

Because the authors open-sourced this through MONAI, you can easily pull the `UNETR` class directly from `monai.networks.nets` in PyTorch without having to code the complex patch-reshaping logic yourself.

---

## SegFormer: Simple and Efﬁcient Design for Semantic Segmentation with Transformers

The SegFormer architecture provides a simplified, highly efficient framework for semantic segmentation by combining a hierarchical Vision Transformer encoder with a lightweight multilayer perceptron (MLP) decoder. By eliminating positional encoding interpolation and complex decoding modules, SegFormer achieves state-of-the-art accuracy while running significantly faster than previous models.

### Patch Splitting and Input 
First, given an input image with dimensions $H \times W \times 3$, you split the image into small patches. SegFormer utilizes patches of size $4 \times 4$, unlike the standard $16 \times 16$ used in typical Vision Transformers (ViT). Smaller patches are critical for semantic segmentation because dense prediction tasks require high-resolution fine features to trace intricate boundaries.

### Overlapped Patch Merging
These patches are processed by the hierarchical Mix Transformer (MiT) encoder to generate multi-scale features. To reduce the resolution at each stage, SegFormer uses an overlapping patch merging process instead of non-overlapping merging. By carefully controlling the patch size $K$, stride $S$, and padding $P$ (e.g., $K=7, S=4, P=3$ or $K=3, S=2, P=1$), the network merges features to resolutions of $\frac{1}{4}, \frac{1}{8}, \frac{1}{16},$ and $\frac{1}{32}$ of the original image while preserving local spatial continuity.

### Efficient Self-Attention
At each stage, the features pass through Transformer blocks. Standard self-attention has a computational complexity of $O(N^2)$, which is highly inefficient for large, high-resolution images. To fix this, SegFormer employs a sequence reduction process: the Key ($K$) sequence length is reduced by a predefined ratio $R$ using reshaping and a linear projection layer. This reduces the length of the sequence to $\frac{N}{R} \times C$, effectively lowering the self-attention complexity to $O(\frac{N^2}{R})$.

### Positional-Encoding-Free Mix-FFN
Standard Transformers use fixed positional encodings (PE), meaning if test resolution differs from training, the PE must be interpolated, causing accuracy drops. SegFormer removes fixed positional encodings entirely. Instead, it uses a Mix Feed-Forward Network (Mix-FFN) that incorporates a $3 \times 3$ depth-wise convolution inside the feed-forward layers. The zero-padding of convolutions naturally leaks location information, allowing the model to adapt perfectly to varying image resolutions during inference.

### The All-MLP Decoder
After the encoder outputs four levels of features (from coarse/high-resolution to fine/low-resolution), they are fed into a lightweight All-MLP decoder. The decoding steps are as follows:
1. **Channel Unification:** Every feature map from the 4 encoder stages passes through an MLP layer to unify their channel dimensions to a uniform size $C$.
2. **Upsampling:** Each unified feature map is upsampled to the exact same spatial resolution (1/4th of the original image).
3. **Fusion:** The upsampled feature maps are concatenated together, and passed through another MLP layer to fuse them.
4. **Prediction:** Finally, a final MLP layer takes these fused features and predicts the semantic segmentation mask at $\frac{H}{4} \times \frac{W}{4} \times N_{cls}$, where $N_{cls}$ is the total number of semantic categories.