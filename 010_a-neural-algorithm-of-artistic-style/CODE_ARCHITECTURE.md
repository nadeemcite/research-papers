# CODE_ARCHITECTURE.md — Neural Style Transfer (Gatys et al.)

## Notebook goal

Produce a self-contained, Colab-runnable notebook that implements the optimisation-based neural style-transfer algorithm from *A Neural Algorithm of Artistic Style*. The notebook loads a pretrained VGG19, extracts content and style targets, and optimises a new image so that it matches the content of one photo and the style of another painting.

## Section-by-section breakdown

### 1. Setup and imports
- Install / import `torch`, `torchvision`, `numpy`, `PIL`, `matplotlib`, `requests`.
- Detect device (`cuda` if available, otherwise `cpu`).
- Set deterministic seeds for reproducibility where possible.

### 2. Download content and style images
- Two helper functions download images from public URLs (or load from local paths if present) and resize them.
- Images are resized to a configurable `IMAGE_SIZE` (default 256 px on the smaller side, keeping aspect ratio; a square crop can also be applied for simplicity).
- We avoid using local files that do not exist on Kaggle; URLs are fetched over the network or a small synthetic image is generated as a fallback.

### 3. Image preprocessing
- Convert PIL images to tensors with `torchvision.transforms.ToTensor()`.
- Normalise with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` so that the pretrained VGG19 receives the expected input distribution.
- Add a batch dimension and move to the device.
- Provide an inverse transform for display: un-normalise, clamp to `[0, 1]`, convert to PIL.

### 4. Load VGG19 and build a feature extractor
- Load `torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)`.
- Keep only the feature extractor (`model.features`) and replace every `MaxPool2d` with `AvgPool2d` to match the paper.
- Freeze all parameters and set to evaluation mode.
- Register forward hooks (or build a small `Sequential` wrapper) to capture activations at:
  - `conv4_2` → content representation
  - `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` → style representation
- Return a dictionary keyed by layer name.

### 5. Content loss
- Compute mean-squared error between the feature map of the generated image and the feature map of the content image at `conv4_2`.
- Store the target content feature once and detach it from the computation graph.
- Formula: `L_content = 0.5 * MSE(F_x, F_p)`.

### 6. Style loss
- For each selected layer, compute the Gram matrix of the generated image features and the target style features.
- Normalise by the number of elements in the feature map (`N * C * H * W` style: `1 / (4 * N_l^2 * M_l^2)` where `M_l = H * W`).
- Weight layers equally (`1 / number_of_layers`) and sum them.
- Formula per layer: `E_l = (1 / (4 N_l^2 M_l^2)) * ||G_x - G_a||_F^2`.

### 7. Total loss and optimisation loop
- Start from the content image (with a small amount of noise) as the initial generated image.
- Use `torch.optim.LBFGS` to minimise `L_total = alpha * L_content + beta * L_style`.
- Use an LBFGS closure that:
  1. zeroes gradients,
  2. runs the feature extractor on the current generated image,
  3. computes content + style loss,
  4. back-propagates,
  5. prints loss values every 50 steps,
  6. clamps pixel values to `[0, 1]` inside the closure to keep a valid image.
- Run for `NUM_STEPS` (default 300; can be reduced on CPU to keep runtime reasonable).

### 8. Visualisation
- Display content image, style image, and generated image side by side with `matplotlib`.
- Optional: show intermediate snapshots during optimisation.

### 9. Reproducibility / runtime notes
- VGG19 weights are downloaded automatically by `torchvision`.
- On a CPU the notebook will run slowly; the Colab/Kaggle GPU flag is enabled in `kernel-metadata.json`.
- Image size is deliberately kept modest to fit free-tier GPU memory and runtime limits.
- If the chosen URLs are unavailable, the notebook falls back to synthetic images so that the code path still executes without errors.

## Key functions / classes

| Name | Role |
|---|---|
| `load_image(url, size)` | Downloads/loads an image and resizes it. |
| `preprocess(img)` | PIL → normalised tensor. |
| `denormalize(tensor)` | Normalised tensor → displayable `[0,1]` image. |
| `VGGFeatures` | Wrapper around VGG19 features with average pooling and forward hooks to capture named-layer activations. |
| `gram_matrix(features)` | Computes the normalised Gram matrix for style representation. |
| `content_loss(gen_features, content_features)` | Squared-error loss at `conv4_2`. |
| `style_loss(gen_features, style_grams)` | Layer-wise Gram-matrix MSE, equally weighted. |
| `run_style_transfer(...)` | Main loop using LBFGS; returns the optimised image tensor. |

## Data flow and tensor shapes

- Input image: `(B=1, C=3, H, W)` after preprocessing.
- VGG19 feature maps:
  - `conv1_1`: `(1, 64, H, W)`
  - `conv2_1`: `(1, 128, H/2, W/2)`
  - `conv3_1`: `(1, 256, H/4, W/4)`
  - `conv4_1`: `(1, 512, H/8, W/8)`
  - `conv4_2`: `(1, 512, H/8, W/8)` (content)
  - `conv5_1`: `(1, 512, H/16, W/16)`
- Gram matrix per layer: `(N_l, N_l)` where `N_l` is the number of filters (64, 128, 256, 512, 512).
- Generated image tensor: same shape as input image; is the trainable variable.

## Deliberate simplifications vs. the full paper

- **No total-variation regulariser.** The original paper did not use TV loss, although many later implementations add it to reduce high-frequency noise. We keep the implementation faithful to the original loss.
- **Smaller image size.** The paper used images around 512 px; we default to 256 px to fit free Colab/Kaggle runtime and memory limits.
- **Content image start instead of white noise.** The paper experimented with both; starting from the content image tends to converge faster and keeps content clearer, so we use it.
- **Equal layer weights for style.** This matches the paper’s convention that each active layer contributes `1 / number_of_layers`.
- **LBFGS optimiser.** We follow the paper and the canonical PyTorch tutorial by using `optim.LBFGS`, which is well suited to small numbers of parameters (the pixels of a single image).
- **Fallback synthetic images.** Kaggle/Colab cannot guarantee external image URLs are reachable, so the notebook uses real URLs when possible and falls back to procedurally generated textures/scenes if downloads fail, ensuring the code path always completes.
