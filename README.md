# Text Line Segmentation of Ancient Handwritten Manuscripts using U-Net

This project focuses on segmenting text lines in images of ancient handwritten manuscripts using a U-Net-based deep learning model. Developed as a Bachelor's Thesis Project (BTP) at IIIT Sricity, the model is trained on the U-DIADS-TL dataset, which includes challenging manuscripts such as Latin2, Latin14396, and Syriaque341. The goal is to accurately extract individual text lines, enabling applications like optical character recognition (OCR) and digital archiving of historical documents.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   https://github.com/KoushikReddy9963/Text-line-segmentation-of-ancient-Handwritten-manuscripts.git
   cd Text-line-segmentation-of-ancient-Handwritten-manuscripts
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8+ installed. Then, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` file includes:
   ```
   opencv-python
   numpy
   tensorflow
   ```

3. **Prepare the Dataset:**
   - Download the U-DIADS-TL dataset and place it in the `U-DIADS-TL` directory.
   - Organize the dataset according to the structure specified in the [Dataset](#dataset) section.

## Usage

### Training the Model
To train the U-Net model from scratch:
1. Ensure the dataset is correctly placed in the `U-DIADS-TL` directory.
2. Run the training script:
   ```bash
   python U-Net_Model.py
   ```
   - This will train the model using the training dataset and save the best model to `text_line_segmentation_model.keras`.
   - Training uses early stopping (patience=10) and model checkpointing to optimize performance.

### Processing a Test Image
To segment text lines in a single test image:
1. Place your test image (e.g., `019.jpg`) in the project directory or specify its path.
2. Modify the `test_image_path` variable in the `if __name__ == "__main__":` block of `U-Net_Model.py` to point to your image:
   ```python
   test_image_path = "./path/to/your/test_image.jpg"
   ```
3. Run the script:
   ```bash
   python U-Net_Model.py
   ```
   - If no trained model exists, it will train one first.
   - The script generates a mask image with segmented text lines, saved as `<original_image_name>_mask.png` (e.g., `019_mask.png`).

## Dataset

The model is trained on the U-DIADS-TL dataset, consisting of 78 images from three ancient manuscripts:
- **Latin2:** Latin-language manuscript with two-column layout and interlinear paratexts.
- **Latin14396:** Latin manuscript with two columns, diverse fonts, and marginal paratexts.
- **Syriaque341:** Syriac manuscript with three columns, vertical comments, and high degradation.

The dataset directory structure should be:
```
U-DIADS-TL/
├── Latin2/
│   ├── img-Latin2/
│   │   ├── training/
│   │   └── validation/
│   └── text-line-gt-Latin2/
│       ├── training/
│       └── validation/
├── Latin14396/
│   ├── img-Latin14396/
│   │   ├── training/
│   │   └── validation/
│   └── text-line-gt-Latin14396/
│       ├── training/
│       └── validation/
└── Syriaque341/
    ├── img-Syriaque341/
    │   ├── training/
    │   └── validation/
    └── text-line-gt-Syriaque341/
        ├── training/
        └── validation/
```

## Model Architecture

The model is based on the U-Net architecture, ideal for image segmentation tasks. It features:
- **Encoder:** 4 convolutional blocks with 3x3 kernels and max-pooling for feature extraction.
- **Bridge:** Convolutional layers with dropout (0.2) to reduce overfitting.
- **Decoder:** Transposed convolutions with skip connections for upsampling and detail retention.
- **Output:** 1x1 convolution with sigmoid activation for pixel-wise segmentation.

A visual representation is provided below:

![U-Net Architecture](https://github.com/KoushikReddy9963/Text-line-segmentation-of-ancient-Handwritten-manuscripts/blob/main/Screenshots/unet-architecture(1).png?raw=true)

For details, see the original U-Net paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

## Results

The model’s performance is demonstrated with sample images from each manuscript:

## Input vs Expected Output vs Model Output : All Languages
- **Latin2:**
  ![Input, Expected Output, Model Output](https://github.com/KoushikReddy9963/Text-line-segmentation-of-ancient-Handwritten-manuscripts/blob/main/Screenshots/1.png?raw=true)


- **Latin14396:**
  ![Input, Expected Output, Model Output](https://github.com/KoushikReddy9963/Text-line-segmentation-of-ancient-Handwritten-manuscripts/blob/main/Screenshots/2.png?raw=true)

- **Syriaque341:**
  ![Input, Expected Output, Model Output](https://github.com/KoushikReddy9963/Text-line-segmentation-of-ancient-Handwritten-manuscripts/blob/main/Screenshots/3.png?raw=true)

The model achieved a pixel-wise accuracy of 42% with a 0.75 threshold, outperforming traditional methods on degraded manuscripts.

## Competition

This project was part of the ICDAR 2025 competition on text line segmentation of ancient manuscripts. Our submission received positive feedback from organizers, with potential inclusion of our names and institute in the official article.

## Contributing

Developed by:
- Yennam Sai Koushik Reddy (S20220010249)
- Karothi Tarun (S20220010099)
- Pederedla Jaswanth (S20220010168)
- Mentor: Dr. Bulla Rajesh

Contributions are welcome! Please submit a pull request or open an issue for suggestions or improvements.

## License

This project is licensed under the MIT License.
