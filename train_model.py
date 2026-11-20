"""Train MobileNetV2 using the original image classes found in dataset/."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
os.environ['KERAS_HOME'] = str(BASE_DIR / '.keras_cache')

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_DIR = BASE_DIR / 'dataset'
LOCAL_TRAIN_CACHE = Path(tempfile.gettempdir()) / 'smart-waste-training-dataset'
MODEL_DIR = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'waste_model.keras'
CLASS_NAMES_PATH = MODEL_DIR / 'class_names.json'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 20


def find_dataset_root(start: Path) -> tuple[Path, list[str]]:
    """Find the first nested directory whose child folders contain images."""
    candidates = []
    for current, dirs, _files in os.walk(start):
        current_path = Path(current)
        classes = []
        for dirname in dirs:
            class_dir = current_path / dirname
            if any(file.suffix.lower() in IMAGE_EXTENSIONS for file in class_dir.iterdir() if file.is_file()):
                classes.append(dirname)
        if len(classes) >= 2:
            image_count = sum(sum(1 for f in (current_path / name).iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS) for name in classes)
            candidates.append((image_count, current_path, sorted(classes)))
    if not candidates:
        raise SystemExit(f'No image category folders found under {start}. Keep the original class folders inside dataset/.')
    _count, root, classes = max(candidates, key=lambda item: item[0])
    return root, classes


def main():
    training_started = time.perf_counter()
    cpu_count = os.cpu_count() or 1
    tf.config.threading.set_intra_op_parallelism_threads(min(4, cpu_count))
    tf.config.threading.set_inter_op_parallelism_threads(min(2, cpu_count))
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset = LOCAL_TRAIN_CACHE if (LOCAL_TRAIN_CACHE / 'garbage_classification').is_dir() else DATA_DIR
    parser.add_argument('--dataset-dir', type=Path, default=default_dataset, help='Folder containing the ZIP extraction')
    args = parser.parse_args()
    dataset_root, class_names = find_dataset_root(args.dataset_dir.resolve())
    print(f'Dataset root: {dataset_root}')
    print(f'Classes ({len(class_names)}): {", ".join(class_names)}')

    train_raw = tf.keras.utils.image_dataset_from_directory(
        dataset_root, class_names=class_names, validation_split=0.2, subset='training', seed=42,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical')
    val_raw = tf.keras.utils.image_dataset_from_directory(
        dataset_root, class_names=class_names, validation_split=0.2, subset='validation', seed=42,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical')

    augmentation = models.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ], name='light_augmentation')

    def prepare(images, labels, training=False):
        if training:
            images = augmentation(images, training=True)
        return preprocess_input(images), labels

    autotune = tf.data.AUTOTUNE
    train_ds = train_raw.map(lambda x, y: prepare(x, y, True), num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_raw.map(lambda x, y: prepare(x, y), num_parallel_calls=autotune).prefetch(autotune)

    backbone = MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    backbone.trainable = False
    # Cache the frozen backbone's image embeddings once. This keeps transfer
    # learning while avoiding a costly MobileNetV2 pass on every training epoch.
    embedding_input = layers.Input(shape=(*IMAGE_SIZE, 3))
    embedding = layers.GlobalAveragePooling2D()(backbone(embedding_input, training=False))
    embedding_model = models.Model(embedding_input, embedding)

    @tf.function(reduce_retracing=True)
    def encode_batch(images):
        return embedding_model(images, training=False)

    def cache_embeddings(dataset, label):
        feature_batches, label_batches = [], []
        total_batches = int(tf.data.experimental.cardinality(dataset).numpy())
        for batch_index, (images, labels) in enumerate(dataset, start=1):
            feature_batches.append(encode_batch(images).numpy())
            label_batches.append(labels.numpy())
            if batch_index % 25 == 0 or batch_index == total_batches:
                print(f'{label.title()} feature extraction: {batch_index}/{total_batches} batches', flush=True)
        print(f'Cached {sum(len(batch) for batch in label_batches)} {label} image embeddings.', flush=True)
        return np.concatenate(feature_batches), np.concatenate(label_batches)

    train_features, train_labels = cache_embeddings(train_ds, 'training')
    val_features, val_labels = cache_embeddings(val_ds, 'validation')

    head_input = layers.Input(shape=(train_features.shape[1],))
    head_hidden = layers.Dropout(0.25)(head_input)
    head_output = layers.Dense(len(class_names), activation='softmax')(head_hidden)
    model = models.Model(head_input, head_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_DIR / 'best_head.keras'), monitor='val_loss', save_best_only=True),
    ]
    model.fit(train_features, train_labels, validation_data=(val_features, val_labels),
              epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)
    train_loss, train_accuracy = model.evaluate(train_features, train_labels, verbose=0)
    val_loss, val_accuracy = model.evaluate(val_features, val_labels, verbose=0)

    # Assemble the trained classification head onto the frozen image backbone.
    image_input = layers.Input(shape=(*IMAGE_SIZE, 3))
    image_features = layers.GlobalAveragePooling2D()(backbone(image_input, training=False))
    full_model = models.Model(image_input, model(image_features))
    full_model.save(MODEL_PATH)
    checkpoint_path = MODEL_DIR / 'best_head.keras'
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    with open(CLASS_NAMES_PATH, 'w', encoding='utf-8') as mapping:
        json.dump(class_names, mapping, indent=2, ensure_ascii=False)
    print(f'Training accuracy: {train_accuracy:.4f}', flush=True)
    print(f'Validation accuracy: {val_accuracy:.4f}', flush=True)
    print(f'Training time: {time.perf_counter() - training_started:.1f} seconds', flush=True)
    print(f'Model saved: {MODEL_PATH}', flush=True)
    print(f'Class mapping saved: {CLASS_NAMES_PATH}', flush=True)


if __name__ == '__main__':
    main()
