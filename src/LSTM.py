# lstm_trainer.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils

class PrintEveryNEpochs(callbacks.Callback):
    """
    Keras callback to print metrics every N epochs.
    """
    def __init__(self, n=10):
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.n == 0 or epoch == 0:
            msg = f"Epoch {epoch + 1}: "
            msg += ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
            print(msg)


class LSTMTrainer:
    """
    Wraps:
      - building an LSTM model for sequence classification
      - training with validation
      - evaluating on test data
      - predicting a class for a single sequence

    Expects:
      X shape: (N, T, D)
      y shape: (N,) integer class labels
    """

    def __init__(self, input_shape, num_classes, lstm_units=128, dropout_rate=0.3, learning_rate=1e-3):
        """
        input_shape: (T, D) - time steps & feature dimension
        num_classes: number of exercise classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = self._build_model()

    def _build_model(self):
        """
        Simple LSTM classifier:
          Input: (T, D)
          LSTM -> Dropout -> Dense(num_classes, softmax)
        """
        inputs = layers.Input(shape=self.input_shape)  # (T, D)

        x = layers.Masking(mask_value=0.0)(inputs)  # in case of any zeros
        x = layers.LSTM(self.lstm_units, return_sequences=False)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())
        return model

    # ---------- utilities for splitting data ----------

    def _train_val_test_split(self, X, y, val_ratio=0.2, test_ratio=0.1, shuffle=True, random_state=42):
        """
        Splits X, y into train/val/test.
        """
        N = X.shape[0]
        indices = np.arange(N)
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(indices)

        X = X[indices]
        y = y[indices]

        test_size = int(N * test_ratio)
        val_size = int(N * val_ratio)

        X_test = X[:test_size]
        y_test = y[:test_size]

        X_val = X[test_size:test_size + val_size]
        y_val = y[test_size:test_size + val_size]

        X_train = X[test_size + val_size:]
        y_train = y[test_size + val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    # ---------- core training API ----------

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, print_every=10):
        """
        Train model on given train (and optional val) data.
        y_* should be integer labels; we convert to one-hot inside.
        """
        # one-hot encode
        y_train_oh = utils.to_categorical(y_train, num_classes=self.num_classes)

        callbacks_list = [PrintEveryNEpochs(n=print_every)]

        validation_data = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            y_val_oh = utils.to_categorical(y_val, num_classes=self.num_classes)
            validation_data = (X_val, y_val_oh)

        history = self.model.fit(
            X_train,
            y_train_oh,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=0  # we control printing via callback
        )
        return history

    def train_val_test(self,X,y, val_ratio=0.2, test_ratio=0.1, epochs=50, batch_size=32, print_every=10):
        """
        Convenience function:
          - splits X,y into train/val/test
          - trains on train+val
          - evaluates on test
        Returns:
          history, test_metrics_dict
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self._train_val_test_split(
            X, y,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        history = self.train(
            X_train, y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            print_every=print_every
        )

        test_loss, test_acc = self.evaluate(X_test, y_test)
        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "X_train_shape": X_train.shape,
            "X_val_shape": X_val.shape,
            "X_test_shape": X_test.shape,
        }
        print("Test metrics:", metrics)
        return history, metrics

    def evaluate(self, X_test, y_test):
        """
        Evaluate on test set.
        """
        if X_test is None or len(X_test) == 0:
            print("No test data provided.")
            return None, None

        y_test_oh = utils.to_categorical(y_test, num_classes=self.num_classes)
        loss, acc = self.model.evaluate(X_test, y_test_oh, verbose=0)
        print(f"Test loss={loss:.4f}, accuracy={acc:.4f}")
        return loss, acc

    # ---------- inference on single sequence ----------

    def predict_sequence(self, seq, label_names=None):
        """
        seq: np.array of shape (T, D) or (1, T, D)

        Returns:
          pred_class_idx, pred_proba, pred_class_name (if label_names given)
        """
        seq = np.array(seq, dtype=np.float32)
        if seq.ndim == 2:
            # (T, D) -> (1, T, D)
            seq = np.expand_dims(seq, axis=0)
        elif seq.ndim != 3:
            raise ValueError(f"Expected seq with shape (T,D) or (1,T,D), got {seq.shape}")

        probs = self.model.predict(seq, verbose=0)[0]  # (num_classes,)
        pred_idx = int(np.argmax(probs))
        pred_proba = float(probs[pred_idx])

        if label_names is not None and 0 <= pred_idx < len(label_names):
            pred_name = label_names[pred_idx]
        else:
            pred_name = None

        return pred_idx, pred_proba, pred_name
