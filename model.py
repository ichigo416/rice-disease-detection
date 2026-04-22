"""
ADSNN-BO Model — Binary Classification (BrownSpot vs Healthy)
Uses pretrained MobileNet ImageNet weights as backbone.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

# ================= CUSTOM PREPROCESS =================
class MobileNetPreprocess(layers.Layer):
    def call(self, x):
        return preprocess_input(x * 255.0)

    def get_config(self):
        return super().get_config()


# ================= CONSTANTS =================
N_CLASSES = 4
IMG_SIZE = 224


# ================= ATTENTION LAYER =================
class AugmentedAttention(layers.Layer):
    def __init__(self, num_heads=4, key_depth=256, value_depth=256, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_depth = key_depth
        self.value_depth = value_depth
        self.total_key_d = num_heads * key_depth
        self.total_val_d = num_heads * value_depth

    def build(self, input_shape):
        fin = input_shape[-1]

        self.Wq = self.add_weight(
            name="Wq", shape=(fin, self.total_key_d), initializer="glorot_uniform"
        )
        self.Wk = self.add_weight(
            name="Wk", shape=(fin, self.total_key_d), initializer="glorot_uniform"
        )
        self.Wv = self.add_weight(
            name="Wv", shape=(fin, self.total_val_d), initializer="glorot_uniform"
        )
        self.Wo = self.add_weight(
            name="Wo", shape=(self.total_val_d, fin), initializer="glorot_uniform"
        )

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.float32)

        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        HW = H * W

        flat = tf.reshape(x, (B, HW, C))

        Q = flat @ self.Wq
        K = flat @ self.Wk
        V = flat @ self.Wv

        def split_heads(t, depth):
            t = tf.reshape(t, (B, HW, self.num_heads, depth))
            return tf.transpose(t, [0, 2, 1, 3])

        Q = split_heads(Q, self.key_depth)
        K = split_heads(K, self.key_depth)
        V = split_heads(V, self.value_depth)

        scale = tf.math.sqrt(tf.cast(self.key_depth, tf.float32))
        scores = tf.matmul(Q, K, transpose_b=True) / scale
        attn = tf.nn.softmax(scores, axis=-1)

        out = tf.matmul(attn, V)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (B, HW, self.total_val_d))
        out = out @ self.Wo
        out = tf.reshape(out, (B, H, W, C))

        return self.layer_norm(x + out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            num_heads=self.num_heads,
            key_depth=self.key_depth,
            value_depth=self.value_depth,
        )
        return cfg


# ================= MODEL BUILDER =================
def build_adsnn_bo(
    input_shape=(224, 224, 3),
    num_heads=4,
    dropout_rate=0.4,
    freeze_base=True,
    no_attention=True,
):

    inputs = layers.Input(shape=input_shape, name="input")

    x = MobileNetPreprocess(name="mobilenet_preprocess")(inputs)

    base = MobileNet(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    base.trainable = not freeze_base

    x = base(x, training=(not freeze_base))

    # Attention (optional)
    if not no_attention:
        x = AugmentedAttention(
            num_heads=num_heads,
            key_depth=1024 // num_heads,
            value_depth=1024 // num_heads,
            name="attention",
        )(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(
        512, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(4, activation="softmax")(x)

    tag = "NoAttn" if no_attention else "WithAttn"
    phase = "Frozen" if freeze_base else "Unfrozen"

    return models.Model(
        inputs, outputs,
        name=f"ADSNN_BO_{tag}_{phase}"
    )


# ================= METRICS =================
def _metrics():
    return [
        "accuracy",
    ]


# ================= TRAIN CONFIG =================
def compile_phase1(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
       loss = "sparse_categorical_crossentropy",
        metrics=_metrics(),
    )
    print(f"  ✅ Phase 1 | backbone=FROZEN | lr={lr}")
    return model


def unfreeze_base(model, finetune_lr=3e-6):
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=finetune_lr,
            clipnorm=1.0,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=_metrics(),
    )
    print(f"  ✅ Phase 2 | backbone=UNFROZEN | lr={finetune_lr}")
    return model


def compile_phase3(model, lr=1e-4):
    for layer in model.layers:
        layer.trainable = "mobilenet_base" not in layer.name

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=1.0,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=_metrics(),
    )
    print(f"  ✅ Phase 3 | attention training | lr={lr}")
    return model
