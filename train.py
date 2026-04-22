import os
import argparse
import warnings 
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight 

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================= GPU SETUP =================
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("✅ GPU detected")
else:
    print("⚠️ Running on CPU")


# ================= IMPORTS =================
from model import build_adsnn_bo, compile_phase1, MobileNetPreprocess, unfreeze_base
from preprocessing import load_dataset, build_tf_dataset, kfold_splits


# ================= HYPERPARAMS =================
HP = {
    "learning_rate": 1e-3,
    "finetune_lr": 1e-5,
    "batch_size": 32,
    "p1_epochs": 15,
    "p2_epochs": 30,
}


# ================= TRAIN FUNCTION =================
def train_fold(X_tr, y_tr, X_val, y_val, fold, output_dir):

    train_ds = build_tf_dataset(X_tr, y_tr, HP["batch_size"], True, True)
    val_ds = build_tf_dataset(X_val, y_val, HP["batch_size"], False, False)

    # Class weights
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight = dict(enumerate(cw))

    print(f"\n=== Fold {fold+1} : Phase 1 ===")

    model = build_adsnn_bo(freeze_base=True, no_attention=True)
    compile_phase1(model, HP["learning_rate"])

    model.fit(
        train_ds,
        epochs=HP["p1_epochs"],
        validation_data=val_ds,
        class_weight=class_weight,
        verbose=1
    )

    # Save Phase 1
    p1_path = output_dir / f"fold{fold}_p1.keras"
    model.save(p1_path)

    print(f"\n=== Fold {fold+1} : Phase 2 ===")

    model_p2 = build_adsnn_bo(freeze_base=False, no_attention=True)

    # Load weights
    p1_model = tf.keras.models.load_model(
        p1_path,
        compile=False,
        custom_objects={"MobileNetPreprocess": MobileNetPreprocess}
    )

    for l1, l2 in zip(model_p2.layers, p1_model.layers):
        try:
            l1.set_weights(l2.get_weights())
        except:
            pass

    # Fine-tune
    unfreeze_base(model_p2, HP["finetune_lr"])

    model_p2.fit(
        train_ds,
        epochs=HP["p2_epochs"],
        validation_data=val_ds,
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate
    results = model_p2.evaluate(val_ds, verbose=0)

    # ✅ FIXED: only 2 outputs now
    val_loss, val_acc = results

    print(f"✅ Fold {fold+1} Accuracy: {val_acc*100:.2f}%")

    return val_acc


# ================= MAIN =================
def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data_dir)

    accs = []

    for fold, X_tr, y_tr, X_val, y_val in kfold_splits(X, y):
        acc = train_fold(X_tr, y_tr, X_val, y_val, fold, output_dir)
        accs.append(acc)

    print("\n🔥 FINAL RESULT")
    print(f"Mean Accuracy: {np.mean(accs)*100:.2f}%")


# ================= RUN ================= 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs")

    args = parser.parse_args()
    main(args) 
