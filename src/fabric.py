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

from src.strategies.trainer_optimizer import *
from src.strategies.loss_functions import *
from src.strategies.activation_function import *
from src.strategies.weight_initializers import *
from src.strategies.classification_strategy import *
from src.strategies.early_stopping import *

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

CLASSIFICATION_STRATEGIES = {
    "argmax": ArgMaxClassification,
    "argmax_random": ArgMaxRandomAtTie,
    "threshold": ThresholdClassification
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
        
    strat_name = config.get("classification_strategy", "argmax")
    if strat_name == "threshold":
        classification_strategy = ThresholdClassification(threshold=config.get("classification_threshold", 0.5))
    else:
        strat_cls = CLASSIFICATION_STRATEGIES[strat_name]
        classification_strategy = strat_cls()

    early_stopping = None
    if config.get("patience") is not None:
        early_stopping = EarlyStopping(
            patience=config["patience"],
            min_delta=config.get("min_delta", 0.0)
        )
        
    return Trainer(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        classification_strategy=classification_strategy,
        learning_rate=config["learning_rate"],
        early_stopping=early_stopping
    )
