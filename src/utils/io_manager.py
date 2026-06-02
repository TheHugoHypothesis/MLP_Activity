"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

class IOManager:
    def __init__(self, base_dir="outputs"):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.reports_dir = os.path.join(base_dir, "reports")
        self.figures_dir = os.path.join(base_dir, "figures")

        os.makedirs(self.base_dir, exist_ok=True)
        self.current_run = None

    def _ensure_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def _make_filename(self, name: str, ext: str):
        return f"{name}.{ext}"

    def _save_json(self, payload: Dict[str, Any], path: str):
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)

    def start_run(self, experiment_id: str) -> str:

        run_name = f"{experiment_id}"
        run_path = os.path.join(self.base_dir, run_name)

        self.models_dir = os.path.join(run_path, "models")
        self.reports_dir = os.path.join(run_path, "reports")
        self.figures_dir = os.path.join(run_path, "figures")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.current_run = run_name
        print(f"[IO] Run directory created: {run_path}")
        return run_name

    def save_experiment_config(self, config: dict, experiment_id: str):
        path = os.path.join(self.reports_dir, self._make_filename(experiment_id, "json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._save_json(config, path)
        print(f"[IO] Configuração do experimento salva em {path}")

    def save_model(self, mlp, experiment_id: str):
        path = os.path.join(self.models_dir, self._make_filename(experiment_id, "json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "layers": []
        }

        for layer in mlp.layers:
            layer_data = []
            for neuron in layer.neurons:
                layer_data.append({
                    "weights": neuron.weights.tolist() if hasattr(neuron.weights, "tolist") else list(neuron.weights),
                    "bias": neuron.bias
                })
            data["layers"].append(layer_data)

        self._save_json(data, path)

        print(f"[IO] Modelo salvo em {path}")

    def save_predictions(self, mlp, dataset: List, experiment_id: str):
        path = os.path.join(self.reports_dir, self._make_filename(experiment_id, "json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        predictions = []
        for sample_index, (x, y) in enumerate(dataset):
            outputs = mlp.forward(x)
            predicted_index = outputs.index(max(outputs))
            target_index = y.index(max(y))

            predictions.append({
                "sample_index": sample_index,
                "target": target_index,
                "predicted": predicted_index,
                "confidence": float(outputs[predicted_index]),
                "correct": predicted_index == target_index,
                "outputs": [float(value) for value in outputs],
            })

        n_samples = len(predictions)
        n_correct = sum(1 for p in predictions if p.get("correct"))
        test_accuracy = n_correct / n_samples if n_samples > 0 else 0.0

        payload = {
            "summary": {
                "test_accuracy": test_accuracy,
                "n_samples": n_samples
            },
            "predictions": predictions
        }

        self._save_json(payload, path)
        print(f"[IO] Saídas da rede salvas em {path}")

    def save_training_history(self, history: dict, experiment_id: str):
        path = os.path.join(self.reports_dir, self._make_filename(experiment_id, "json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._save_json(history, path)
        print(f"[IO] Histórico de treinamento salvo em {path}")

    def load_model(self, mlp_class, experiment_id: str, *args, **kwargs):
        path = os.path.join(self.models_dir, self._make_filename(experiment_id, "json"))

        with open(path, "r") as f:
            data = json.load(f)

        mlp = mlp_class(*args, **kwargs)

        for i, layer in enumerate(data["layers"]):
            for j, neuron_data in enumerate(layer):
                neuron = mlp.layers[i].neurons[j]
                neuron.weights = neuron_data["weights"]
                neuron.bias = neuron_data["bias"]

        print(f"[IO] Modelo carregado de {path}")
        return mlp

    def save_report(self, report: dict, experiment_id: str):
        path = os.path.join(self.reports_dir, self._make_filename(experiment_id, "json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._save_json(report, path)

        print(f"[IO] Relatório salvo em {path}")