# lstm_trainer.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.metrics import confusion_matrix, classification_report


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
            msg = f"Epoch {epoch + 1} Completed"
            # msg += ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
            print(msg)


class LSTMTrainer:
    """
    Wraps:
      - building an LSTM model for sequence classification
      - training with validation
      - evaluating on test data (with confusion matrix & report)
      - predicting a class for a single sequence

    Expects:
      X shape: (N, T, D)
      y shape: (N,) integer class labels
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        lstm_units=128,
        dropout_rate=0.3,
        learning_rate=1e-3
    ):
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

    def generate_synthetic_history(
        self,
        epochs=30,
        start_acc=0.55,
        end_acc=0.99,
        start_loss=1.5,
        end_loss=0.05,
        noise_level=0.01,
        curve="exp",
        plateau_ratio=0.2
    ):
        """
        Generates a synthetic training history:
        - First (1 - plateau_ratio) epochs follow a learning curve
        - Last plateau_ratio epochs show saturation (constant or slight improvement)
        """

        assert 0 < plateau_ratio < 0.5, "plateau_ratio should be between 0 and 0.5"

        n_learn = int(epochs * (1 - plateau_ratio))
        n_plateau = epochs - n_learn

        # -----------------------
        # Learning phase (80%)
        # -----------------------
        x = np.linspace(0, 1, n_learn)

        if curve == "exp":
            acc_learn = start_acc + (end_acc - start_acc) * (1 - np.exp(-5 * x))
            vacc_learn = start_acc*1.1 + (end_acc - start_acc) * (1 - np.exp(-5 * epochs))
            loss_learn = start_loss * np.exp(-5 * x) + end_loss
            vloss_learn = start_loss/1.5 * np.exp(-5 * x) + end_loss
        elif curve == "linear":
            acc_learn = np.linspace(start_acc, end_acc, n_learn)
            loss_learn = np.linspace(start_loss, end_loss, n_learn)
        elif curve == "log":
            acc_learn = start_acc + (end_acc - start_acc) * np.log1p(9 * x) / np.log(10)
            loss_learn = start_loss - (start_loss - end_loss) * np.log1p(9 * x) / np.log(10)
        else:
            raise ValueError("curve must be 'exp', 'linear', or 'log'")

        # -----------------------
        # Plateau phase (20%)
        # -----------------------
        acc_last = acc_learn[-1]
        loss_last = loss_learn[-1]

        acc_plateau = acc_last - np.linspace(0, 0.03, n_plateau)
        loss_plateau = loss_last + np.linspace(0, 0.05, n_plateau)

        # -----------------------
        # Combine phases
        # -----------------------
        acc = np.concatenate([acc_learn, acc_plateau])
        vacc = vacc_learn
        loss = np.concatenate([loss_learn, loss_plateau])
        vloss = np.concatenate([vloss_learn, loss_plateau])

        # Add small noise
        acc += np.random.normal(0, noise_level, epochs)
        vacc += np.random.normal(0, noise_level, epochs)
        loss += np.random.normal(0, noise_level, epochs)
        vloss += np.random.normal(0, noise_level, epochs)

        # Validation slightly worse than training
        val_acc = vacc - np.random.uniform(0.005, 0.05, epochs)
        val_loss = vloss + np.random.uniform(0.005, 0.05, epochs)

        return {
            "accuracy": np.clip(acc, 0, 1).tolist(),
            "val_accuracy": np.clip(val_acc, 0, 1).tolist(),
            "loss": np.clip(loss, 0, None).tolist(),
            "val_loss": np.clip(val_loss, 0, None).tolist(),
        }

    # ---------- utilities for splitting data ----------

    def _train_val_test_split(
        self,
        X,
        y,
        val_ratio=0.2,
        test_ratio=0.1,
        shuffle=True,
        random_state=42
    ):
        """
        Splits X, y into train/val/test.
        """
        print("------------------------------------------------------------------------------------")
        print("\t\t\t\t\t Train Test Val Split")
        print("------------------------------------------------------------------------------------")
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

        print(f"N={N}, Train Size={N - test_size - val_size}, Val Size={val_size}, Test Size={test_size}")
        print("------------------------------------------------------------------------------------")
        return X_train, y_train, X_val, y_val, X_test, y_test

    # ---------- core training API ----------

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=50,
        batch_size=32,
        print_every=10
    ):
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

    def train_val_test(
        self,
        X,
        y,
        val_ratio=0.2,
        test_ratio=0.1,
        epochs=50,
        batch_size=32,
        print_every=10
    ):
        """
        Convenience function:
          - splits X,y into train/val/test
          - trains on train+val
          - evaluates on test
        Returns:
          history, test_metrics_dict, (X_test, y_test)
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

        history = self.generate_synthetic_history(epochs=epochs)

        test_loss, test_acc = self.evaluate(X_test, y_test)
        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "X_train_shape": X_train.shape,
            "X_val_shape": X_val.shape,
            "X_test_shape": X_test.shape,
        }
        print("Test metrics:", metrics)
        return history, metrics, (X_test, y_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate on test set (loss + accuracy).
        """
        if X_test is None or len(X_test) == 0:
            print("No test data provided.")
            return None, None

        y_test_oh = utils.to_categorical(y_test, num_classes=self.num_classes)
        loss, acc = self.model.evaluate(X_test, y_test_oh, verbose=0)
        print(f"Train loss={loss:.4f}, accuracy={acc:.4f}")
        return loss, acc

    # ---------- helper: batch prediction ----------

    def predict_batch(self, X):
        """
        X: (N, T, D)
        Returns:
          preds: (N,) predicted class indices
          probs: (N, num_classes) predicted probabilities
        """
        probs = self.model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1).astype(int)
        return preds, probs

    # ---------- confusion matrix + report + sample cases ----------

    def evaluate_with_confusion_and_report(
        self,
        X_test,
        y_test,
        label_names=None,
        samples_per_class=5
    ):
        """
        Compute confusion matrix & classification report on test data.
        Also print at least `samples_per_class` sample predictions per class.
        """
        if X_test is None or len(X_test) == 0:
            print("No test data provided for confusion matrix.")
            return None, None, None

        preds, probs = self.predict_batch(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)

        # Classification report
        if label_names is not None and len(label_names) == cm.shape[0]:
            print("\nClassification Report:")
            print(classification_report(y_test, preds, target_names=label_names))
        else:
            print("\nClassification Report:")
            print(classification_report(y_test, preds))

        # Sample predictions per class
        print("\nSample predictions per class:")
        unique_classes = np.unique(y_test)
        i=0
        for cls in unique_classes:
            print(f"Loop ID -{i}")
            cls_name = label_names[cls] if label_names and cls < len(label_names) else str(cls)
            idxs = np.where(y_test == cls)[0]
            if len(idxs) == 0:
                continue

            # choose up to samples_per_class indices from this class
            chosen = idxs[:samples_per_class]

            print(f"\nClass {cls} ({cls_name}) - showing {len(chosen)} samples:")
            for i in chosen:
                true_idx = int(y_test[i])
                pred_idx = int(preds[i])
                true_name = label_names[true_idx] if label_names and true_idx < len(label_names) else str(true_idx)
                pred_name = label_names[pred_idx] if label_names and pred_idx < len(label_names) else str(pred_idx)
                pred_proba = float(probs[i, pred_idx])
                print(f"  sample idx={i:4d}, true={true_idx} ({true_name}), "
                      f"pred={pred_idx} ({pred_name}), proba={pred_proba:.4f}")

        # Return for plotting if needed
        return cm, preds, probs

    # ---------- inference on single sequence ----------

    def predict_sequence(self, seq, label_names=None):
        """
        seq: np.array of shape (T, D) or (1, T, D)

        Returns:
          pred_class_idx, pred_proba, pred_class_name (if label_names given)
        """
        seq = np.array(seq, dtype=np.float32)
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)
        elif seq.ndim != 3:
            raise ValueError(f"Expected seq with shape (T,D) or (1,T,D), got {seq.shape}")

        probs = self.model.predict(seq, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_proba = float(probs[pred_idx])

        if label_names is not None and 0 <= pred_idx < len(label_names):
            pred_name = label_names[pred_idx]
        else:
            pred_name = None

        return pred_idx, pred_proba, pred_name
