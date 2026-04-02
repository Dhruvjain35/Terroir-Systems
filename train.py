#!/usr/bin/env python3
"""
Terroir AI — Model Training
============================
Run AFTER downloading PlantVillage dataset from Kaggle.
https://www.kaggle.com/datasets/arjuntejaswi/plant-village

  python train.py --cnn   → MobileNetV2 → INT8 TFLite (3x CPU speedup)
  python train.py --svm   → SIFT + BoVW + SVM RBF kernel
  python train.py --both  → train both

Data layout expected:
  data/train/Healthy/         ← from PlantVillage
  data/train/Early_Blight/    ← from PlantVillage
  data/train/Late_Blight/     ← from PlantVillage
  data/svm/Healthy/           ← custom structural defect images
  data/svm/Dry_Rot/
  data/svm/Gangrene/
"""

import os, sys, json, pickle, argparse
import numpy as np
import cv2

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CNN_CLASSES = ["Healthy", "Early_Blight", "Late_Blight"]
SVM_CLASSES = ["Healthy", "Dry_Rot", "Gangrene"]


# ══════════════════════════════════════════════════════════════════════
#  CNN: MobileNetV2 → INT8 TFLite
#  Spec: INT8 quantization reduces model 75%, speeds CPU inference 3x
# ══════════════════════════════════════════════════════════════════════

def train_cnn(data_dir="data", epochs=25, batch_size=32):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        print("ERROR: pip install tensorflow"); return

    IMG = 224
    print(f"\n{'='*55}")
    print("  CNN Training: MobileNetV2 → INT8 TFLite")
    print(f"{'='*55}")
    print(f"  Classes: {CNN_CLASSES}")
    print(f"  Epochs:  {epochs}, Batch: {batch_size}\n")

    gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True, vertical_flip=True,
        rotation_range=30, zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        width_shift_range=0.1, height_shift_range=0.1,
        validation_split=0.2,
    )
    train_ds = gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(IMG,IMG), batch_size=batch_size,
        class_mode="categorical", subset="training", classes=CNN_CLASSES,
    )
    val_ds = gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(IMG,IMG), batch_size=batch_size,
        class_mode="categorical", subset="validation", classes=CNN_CLASSES,
    )

    # MobileNetV2 base (lightweight — designed for edge inference)
    base = MobileNetV2(input_shape=(IMG,IMG,3), include_top=False, weights="imagenet")
    base.trainable = False

    # Custom head — 3×3 kernels per paper spec, GAP reduces params
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.30),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(len(CNN_CLASSES), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    cb = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR,"cnn_best.h5"),
            save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    ]

    print("  Phase 1: head training (base frozen)...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs//2, callbacks=cb)

    print("  Phase 2: fine-tuning (top 30 base layers unfrozen)...")
    base.trainable = True
    for layer in base.layers[:-30]: layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs//2, callbacks=cb)

    # ── INT8 Quantization (spec: 75% smaller, 3x faster) ──────────────
    print("\n  Converting to INT8 TFLite...")
    def rep_data():
        for _ in range(200):
            yield [np.random.rand(1,IMG,IMG,3).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.uint8
    conv.inference_output_type = tf.uint8
    tfl = conv.convert()

    out = os.path.join(MODEL_DIR, "cnn_model.tflite")
    with open(out, "wb") as f: f.write(tfl)
    with open(os.path.join(MODEL_DIR,"cnn_classes.json"),"w") as f:
        json.dump(CNN_CLASSES, f)

    mb = len(tfl)/(1024*1024)
    print(f"  ✓ Saved: {out} ({mb:.1f}MB INT8)")
    print(f"  Classes: {CNN_CLASSES}\n")


# ══════════════════════════════════════════════════════════════════════
#  SIFT-SVM: Scale-Invariant Features + RBF Support Vector Machine
#  Spec: for structural defects (Dry Rot) — invariant to potato tumbling
#  Math: D(x,y,σ) = (G(x,y,kσ)-G(x,y,σ))*I(x,y)   (DoG keypoints)
#        SVM: min ||w||²/2  s.t. y_i(w·x_i+b) ≥ 1   (RBF kernel)
# ══════════════════════════════════════════════════════════════════════

def train_svm(data_dir="data/svm", n_clusters=50):
    try:
        from sklearn.svm import SVC
        from sklearn.cluster import KMeans
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("ERROR: pip install scikit-learn"); return

    print(f"\n{'='*55}")
    print("  SIFT-SVM Training (Bag-of-Visual-Words + RBF SVM)")
    print(f"{'='*55}")

    sift    = cv2.SIFT_create()
    IMG     = 384   # paper minimum
    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir,d))])
    print(f"  Classes: {classes}\n")

    # ── 1. Extract SIFT descriptors ───────────────────────────────────
    all_desc   = []
    per_image  = []   # (descriptors, label_idx)

    for li, cls in enumerate(classes):
        d = os.path.join(data_dir, cls)
        imgs = [f for f in os.listdir(d) if f.lower().endswith((".jpg",".jpeg",".png"))]
        print(f"  {cls}: {len(imgs)} images")
        for fn in imgs:
            img = cv2.imread(os.path.join(d,fn), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (IMG,IMG))
            kp, des = sift.detectAndCompute(img, None)
            if des is not None and len(des)>=4:
                all_desc.append(des)
                per_image.append((des, li))

    if not all_desc:
        print("ERROR: no descriptors extracted — check data directory"); return

    # ── 2. K-Means vocabulary (BoVW) ─────────────────────────────────
    print(f"\n  Building vocabulary ({n_clusters} visual words)...")
    flat  = np.vstack(all_desc)
    km    = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(flat)

    # ── 3. Encode each image as histogram over vocabulary ─────────────
    print("  Encoding images as BoVW histograms...")
    X, y = [], []
    for des, li in per_image:
        dists  = np.linalg.norm(des[:,None] - km.cluster_centers_[None,:], axis=2)
        assign = np.argmin(dists, axis=1)
        hist   = np.bincount(assign, minlength=n_clusters).astype(float)
        hist  /= hist.sum()+1e-8
        X.append(hist); y.append(li)
    X = np.array(X); y = np.array(y)

    # ── 4. RBF SVM ────────────────────────────────────────────────────
    print("  Training SVM (RBF kernel)...")
    pipe = Pipeline([("sc", StandardScaler()),
                     ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                                 probability=True))])
    cv   = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"  Cross-val: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
    pipe.fit(X, y)

    # Attach vocab + classes for inference lookup
    pipe.vocabulary_ = km.cluster_centers_
    pipe.classes_     = classes

    out = os.path.join(MODEL_DIR,"sift_svm.pkl")
    with open(out,"wb") as f: pickle.dump(pipe, f)
    print(f"  ✓ Saved: {out}  ({cv.mean()*100:.1f}% accuracy)")
    print(f"  Classes: {classes}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cnn",  action="store_true")
    p.add_argument("--svm",  action="store_true")
    p.add_argument("--both", action="store_true")
    p.add_argument("--data", default="data")
    a = p.parse_args()

    if a.both or a.cnn:  train_cnn(a.data)
    if a.both or a.svm:  train_svm(os.path.join(a.data,"svm"))
    if not (a.both or a.cnn or a.svm):
        print("Usage: python train.py --cnn | --svm | --both [--data DIR]")
