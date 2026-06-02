"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

from src.core.network import MultilayerPerceptron
from src.core.trainer import Trainer
from src.core.layer import LayerConfig
from src.core.matrix_network import MatrixMultilayerPerceptron
from src.core.matrix_trainer import MatrixTrainer

from src.strategies.trainer_optimizer import *
from src.strategies.loss_functions import *
from src.strategies.activation_function import *
from src.strategies.weight_initializers import *

""" Dicionário de estratégias que podem ser utilizadas """
ACTIVATIONS = {
    "relu": RELU,
    "leaky_relu": LeakyRELU,
    "sigmoid": Sigmoid,
    "linear": Linear
}

INITIALIZERS = {
    "he": HeInitializer,
    "xavier": XavierGlorotInitializer,
    "uniform": UniformInitializer,
    "normal": NormalInitializer
}

LOSS_FUNCTIONS = {
    "softmax_cross_entropy": SoftmaxCrossEntropy,
    "mse": MSE,
    "mae": MAE
}

OPTIMIZERS = {
    "sgd": SGD,
    "sgd_momentum": SGD_momentum
}

""" Construtores de treinador e modelo """
def build_model(config: dict):
    layer_configs = []
    for layer in config["layers"]:
        activation_cls = ACTIVATIONS[layer["activation"]]
        initializer_cls = INITIALIZERS[layer["initializer"]]
        
        layer_configs.append(
            LayerConfig(
                n_neurons=layer["n_neurons"],
                activation=activation_cls(),
                initializer=initializer_cls()
            )
        )
        
    if config.get("use_matrix_vectorization", False):
        return MatrixMultilayerPerceptron(
            layer_configs=layer_configs,
            input_size=config["input_size"]
        )
        
    return MultilayerPerceptron(
        layer_configs=layer_configs,
        input_size=config["input_size"],
        use_numpy=config.get("use_numpy", False)
    )

def build_trainer(model, config: dict):
    loss_cls = LOSS_FUNCTIONS[config["loss_function"]]
    loss_fn = loss_cls()
    opt_config = config["optimizer"]
    if opt_config["type"] == "sgd_momentum":
        optimizer = SGD_momentum(
            momentum=opt_config.get("momentum", 0.9),
            l2_decay=opt_config.get("l2_decay", 0.0)
        )
    else:
        optimizer = SGD()
        
    if config.get("use_matrix_vectorization", False):
        return MatrixTrainer(
            model=model,
            loss_function=loss_fn,
            optimizer=optimizer,
            learning_rate=config["learning_rate"],
            patience=config.get("patience"),
            min_delta=config.get("min_delta", 0.0)
        )
        
    return Trainer(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        learning_rate=config["learning_rate"],
        patience=config.get("patience"),
        min_delta=config.get("min_delta", 0.0)
    )
