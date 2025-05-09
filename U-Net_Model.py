import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = 512
CHANNELS = 1
BATCH_SIZE = 4
MODEL_PATH = "C:/BTP/text_line_segmentation_model.keras"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

DATA_PATHS = {
    'train': {
        'Latin2': './U-DIADS-TL/Latin2/img-Latin2/training',
        'Latin14396': './U-DIADS-TL/Latin14396/img-Latin14396/training',
        'Syriaque341': './U-DIADS-TL/Syriaque341/img-Syriaque341/training'
    },
    'val': {
        'Latin2': './U-DIADS-TL/Latin2/img-Latin2/validation',
        'Latin14396': './U-DIADS-TL/Latin14396/img-Latin14396/validation',
        'Syriaque341': './U-DIADS-TL/Syriaque341/img-Syriaque341/validation'
    }
}

GROUND_TRUTH_DATA_PATHS = {
    'train': {
        'Latin2': './U-DIADS-TL/Latin2/text-line-gt-Latin2/training',
        'Latin14396': './U-DIADS-TL/Latin14396/text-line-gt-Latin14396/training',
        'Syriaque341': './U-DIADS-TL/Syriaque341/text-line-gt-Syriaque341/training'
    },
    'val': {
        'Latin2': './U-DIADS-TL/Latin2/text-line-gt-Latin2/validation',
        'Latin14396': './U-DIADS-TL/Latin14396/text-line-gt-Latin14396/validation',
        'Syriaque341': './U-DIADS-TL/Syriaque341/text-line-gt-Syriaque341/validation'
    }
}

def build_unet():
    """Build U-Net architecture for text line segmentation"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    
    # Encoder
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bridge
    c4 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c7)
    
    return models.Model(inputs=inputs, outputs=outputs)

def create_dataset(data_type='train'):
    """Create segmentation dataset with images and masks"""
    file_paths = []
    mask_paths = []
    
    for class_name, dir_path in DATA_PATHS[data_type].items():
        mask_dir = GROUND_TRUTH_DATA_PATHS[data_type][class_name]
        for img_file in os.listdir(dir_path):
            if not img_file.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
                
            img_path = os.path.join(dir_path, img_file)
            img_name = os.path.splitext(img_file)[0]
            mask_file = f"{img_name}.png"
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                file_paths.append(img_path)
                mask_paths.append(mask_path)

    def load_and_preprocess(img_path, mask_path):
        def _process(img_path, mask_path):
            img = cv2.imread(img_path.decode('utf-8'), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Invalid image path: {img_path}")
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            
            mask = cv2.imread(mask_path.decode('utf-8'), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Invalid mask path: {mask_path}")
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            mask = (mask > 127).astype(np.float32)
            
            return img[..., np.newaxis], mask[..., np.newaxis]
        
        img, mask = tf.numpy_function(_process, [img_path, mask_path], (tf.float32, tf.float32))
        img.set_shape((IMG_SIZE, IMG_SIZE, CHANNELS))
        mask.set_shape((IMG_SIZE, IMG_SIZE, 1))
        return img, mask
    
    return tf.data.Dataset.from_tensor_slices((file_paths, mask_paths)) \
                         .shuffle(100) \
                         .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                         .batch(BATCH_SIZE) \
                         .prefetch(tf.data.AUTOTUNE)

def train_model():
    """Train and save the segmentation model"""
    print("Starting model training...")
    train_dataset = create_dataset('train')
    val_dataset = create_dataset('val')
    
    model = build_unet()
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])])
    
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, save_best_only=True)
        ]
    )
    print("Model training completed and saved!")
    return model

def process_test_image(test_image_path):
    """Process a single test image and generate its mask"""
    # Check if model exists, if not train it
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Training new model...")
        train_model()
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Load and preprocess test image
    original_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        raise ValueError(f"Could not read image from path: {test_image_path}")
    
    # Resize and normalize
    img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    
    # Predict mask
    pred_mask = model.predict(np.expand_dims(img[..., np.newaxis], axis=0))[0]
    
    # Post-processing
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask.squeeze(), (original_img.shape[1], original_img.shape[0]))
    
    # Create final output
    final_mask = np.zeros_like(original_img)
    final_mask[pred_mask == 255] = 255
    
    # Generate output path from input path
    output_path = os.path.splitext(test_image_path)[0] + "_mask.png"
    
    # Save result
    cv2.imwrite(output_path, final_mask)
    print(f"Saved text line mask to: {output_path}")
    return final_mask

if __name__ == "__main__":
    # Process single test image
    test_image_path = "./019.jpg"  # Replace with your test image path
    process_test_image(test_image_path)