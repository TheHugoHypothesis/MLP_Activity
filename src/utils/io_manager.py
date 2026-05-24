import json
import os
from datetime import datetime


class IOManager:
    def __init__(self, base_dir="outputs"):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.reports_dir = os.path.join(base_dir, "reports")
        self.figures_dir = os.path.join(base_dir, "figures")

    def _make_filename(self, name: str, ext: str):
        return f"{name}.{ext}"

    def save_model(self, mlp, experiment_id: str):
        path = os.path.join(self.models_dir, self._make_filename(experiment_id, "json"))

        data = {
            "layers": []
        }

        for layer in mlp.layers:
            layer_data = []
            for neuron in layer.neurons:
                layer_data.append({
                    "weights": neuron.weight_list,
                    "bias": neuron.bias
                })
            data["layers"].append(layer_data)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[IO] Modelo salvo em {path}")

    def load_model(self, mlp_class, experiment_id: str, *args, **kwargs):
        path = os.path.join(self.models_dir, self._make_filename(experiment_id, "json"))

        with open(path, "r") as f:
            data = json.load(f)

        mlp = mlp_class(*args, **kwargs)

        for i, layer in enumerate(data["layers"]):
            for j, neuron_data in enumerate(layer):
                neuron = mlp.layers[i].neurons[j]
                neuron.weight_list = neuron_data["weights"]
                neuron.bias = neuron_data["bias"]

        print(f"[IO] Modelo carregado de {path}")
        return mlp

    def save_report(self, report: dict, experiment_id: str):
        path = os.path.join(self.reports_dir, self._make_filename(experiment_id, "json"))

        with open(path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"[IO] Relatório salvo em {path}")