"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from typing import List
import numpy as np

class MatrixTrainer:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        learning_rate: float = 0.01,
        patience: int = None,
        min_delta: float = 0.0
    ):
        self.model = model
        self.loss_function = loss_function
        self.loss_fn_name = loss_function.__class__.__name__
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
        
        self.v_W = [np.zeros_like(w) for w in self.model.W]
        self.v_b = [np.zeros_like(b) for b in self.model.b]
        
        opt_name = optimizer.__class__.__name__.lower()
        if "momentum" in opt_name:
            self.opt_type = "sgd_momentum"
        else:
            self.opt_type = "sgd"
            
        self.momentum = getattr(optimizer, "momentum", 0.9)
        self.l2_decay = getattr(optimizer, "l2_decay", 0.0)

    def update_weights(self):
        for l in range(len(self.model.W)):
            grad_W = self.model.dW[l]
            if self.l2_decay > 0.0:
                grad_W = grad_W + self.l2_decay * self.model.W[l]
                
            if self.opt_type == "sgd_momentum":
                self.v_W[l] = self.momentum * self.v_W[l] + self.learning_rate * grad_W
                self.model.W[l] -= self.v_W[l]
                
                self.v_b[l] = self.momentum * self.v_b[l] + self.learning_rate * self.model.db[l]
                self.model.b[l] -= self.v_b[l]
            else:
                self.model.W[l] -= self.learning_rate * grad_W
                self.model.b[l] -= self.learning_rate * self.model.db[l]

    def compute_loss_and_acc(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
        Y_pred = self.model.forward(X)
        
        if self.loss_fn_name == "SoftmaxCrossEntropy":
            max_logit = np.max(Y_pred, axis=0, keepdims=True)
            exp_vals = np.exp(Y_pred - max_logit)
            probs = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
            clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
            loss = np.mean(-np.sum(Y * np.log(clipped), axis=0))
        elif self.loss_fn_name == "MSE":
            loss = np.mean(np.sum((Y - Y_pred) ** 2, axis=0) / Y_pred.shape[0])
        elif self.loss_fn_name == "MAE":
            loss = np.mean(np.sum(np.abs(Y - Y_pred), axis=0) / Y_pred.shape[0])
            
        pred_labels = np.argmax(Y_pred, axis=0)
        true_labels = np.argmax(Y, axis=0)
        accuracy = np.mean(pred_labels == true_labels)
        
        return float(loss), float(accuracy)

    def evaluate(self, dataset: List) -> tuple[float, float]:
        if not dataset:
            return 0.0, 0.0
        X = np.array([x for x, y in dataset]).T
        Y = np.array([y for x, y in dataset]).T
        return self.compute_loss_and_acc(X, Y)

    def train(
        self,
        train_dataset: List,
        val_dataset: List,
        epochs: int
    ):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        
        X_train = np.array([x for x, y in train_dataset]).T
        Y_train = np.array([y for x, y in train_dataset]).T
        
        X_val = np.array([x for x, y in val_dataset]).T if val_dataset else None
        Y_val = np.array([y for x, y in val_dataset]).T if val_dataset else None
        
        precomputed_train = [
            (np.array(x, dtype=np.float64).reshape(-1, 1), np.array(y, dtype=np.float64).reshape(-1, 1))
            for x, y in train_dataset
        ]
        
        if self.loss_fn_name == "SoftmaxCrossEntropy":
            def get_dY(Y_p, Y_s):
                max_logit = np.max(Y_p, axis=0, keepdims=True)
                exp_vals = np.exp(Y_p - max_logit)
                probs = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
                return probs - Y_s
        elif self.loss_fn_name == "MSE":
            scale = 2.0 / Y_train.shape[0]
            def get_dY(Y_p, Y_s):
                return scale * (Y_p - Y_s)
        elif self.loss_fn_name == "MAE":
            scale = 1.0 / Y_train.shape[0]
            def get_dY(Y_p, Y_s):
                return scale * np.sign(Y_p - Y_s)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights_snapshot = None

        for epoch in range(epochs):
            for X_sample, Y_sample in precomputed_train:
                Y_pred = self.model.forward(X_sample)
                dY = get_dY(Y_pred, Y_sample)
                self.model.backward(dY)
                self.update_weights()
                
            average_train_loss, average_train_acc = self.compute_loss_and_acc(X_train, Y_train)
            
            if X_val is not None:
                average_val_loss, average_val_acc = self.compute_loss_and_acc(X_val, Y_val)
            else:
                average_val_loss, average_val_acc = 0.0, 0.0
                
            history["train_loss"].append(average_train_loss)
            history["val_loss"].append(average_val_loss)
            history["train_acc"].append(average_train_acc)
            history["val_acc"].append(average_val_acc)
            
            print(
                f"Época {epoch} | "
                f"Train Loss: {average_train_loss:.6f} | "
                f"Val Loss: {average_val_loss:.6f} | "
                f"Val Acc: {average_val_acc:.4f}"
            )
            
            if self.patience is not None:
                if average_val_loss < (best_val_loss - self.min_delta):
                    best_val_loss = average_val_loss
                    patience_counter = 0
                    best_weights_snapshot = self._get_model_weights_snapshot()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\nEarly stop na época {epoch}.")
                        print(f"O erro de validação não melhorou por {self.patience} épocas seguidas.")
                        self._restore_model_weights(best_weights_snapshot)
                        print("[Early Stopping] Melhores pesos restaurados.")
                        break
                        
        return history

    def _get_model_weights_snapshot(self):
        return [
            (np.copy(W_l), np.copy(b_l))
            for W_l, b_l in zip(self.model.W, self.model.b)
        ]
        
    def _restore_model_weights(self, snapshot):
        for l, (W_snap, b_snap) in enumerate(snapshot):
            self.model.W[l] = np.copy(W_snap)
            self.model.b[l] = np.copy(b_snap)
