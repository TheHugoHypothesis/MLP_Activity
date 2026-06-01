"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

import time

""" Tempo para a execução de códigos """
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def exibir_dashboard_tempos(times: dict, n_epochs: int = None):
    inner_w = 56
    def border(left, mid, right): return f"{left}{mid * (inner_w + 2)}{right}"
    def row(text): return f"│ {text:<{inner_w}} │"
    def center_row(text): return f"│{text.center(inner_w + 2)}│"
    lines = [
        border("┌", "─", "┐"),
        center_row("TEMPOS DE EXECUÇÃO DO PIPELINE"),
        border("├", "─", "┤")
    ]
    
    total_time = sum(times.values())
    
    lines.append(row(f"Carga do Dataset (DataLoader):    {times.get('load_data', 0.0):.4f}s"))
    
    train_time = times.get('train', 0.0)
    lines.append(row(f"Treinamento do Modelo (Trainer):   {train_time:.4f}s"))
    
    if n_epochs is not None and n_epochs > 0:
        time_per_epoch = train_time / n_epochs
        lines.append(row(f"Tempo médio por Época:         {time_per_epoch:.4f}s/época ({n_epochs} épocas)"))
        
    lines.append(border("├", "─", "┤"))
    lines.append(row(f"Tempo Total Cronometrado:          {total_time:.4f}s"))
    lines.append(border("└", "─", "┘\n"))
    
    print("\n".join(lines))

def exibir_dashboard_configuracoes(config: dict):
    inner_w = 56
    def border(left, mid, right): return f"{left}{mid * (inner_w + 2)}{right}"
    def row(text): return f"│ {text:<{inner_w}} │"
    def center_row(text): return f"│{text.center(inner_w + 2)}│"
    lines = [
        border("┌", "─", "┐"),
        center_row("CONFIGURAÇÃO INICIAL DO EXPERIMENTO"),
        border("├", "─", "┤"),
        row(f"Experimento:   {config['experiment_id']}")
    ]
    
    modo = "Cross-Validation" if config["use_cross_validation"] else "Holdout"
    if config["use_cross_validation"]:
        detalhe_modo = f"{modo} ({config['cross_validation_folds']} Folds)"
    else:
        detalhe_modo = f"{modo} ({config['hold_out_p_train']*100:.1f}% Treino, {config['hold_out_p_validation']*100:.1f}% Val)"
    
    lines.append(row(f"Modo:          {detalhe_modo}"))
    lines.append(row(f"Épocas:        {config['num_epochs']}"))
    lines.append(row(f"Learning Rate: {config['learning_rate']}"))
    lines.append(row(f"Early Stop:    Paciência: {config['patience']} | Delta Mínimo: {config['min_delta']}"))
    
    lines.append(border("├", "─", "┤"))
    lines.append(center_row("ARQUITETURA DA REDE"))
    lines.append(border("├", "─", "┤"))
    lines.append(row(f"Entrada:       {config['input_size']} entradas"))
    
    for idx, layer in enumerate(config["layers"], start=1):
        desc = f"Camada {idx}:      {layer['n_neurons']} neurônios | {layer['activation']} | {layer['initializer']}"
        lines.append(row(desc))
        
    lines.append(border("├", "─", "┤"))
    lines.append(center_row("ESTRATÉGIAS DE TREINAMENTO"))
    lines.append(border("├", "─", "┤"))
    
    lines.append(row(f"Perda:         {config['loss_function']}"))
    opt = config["optimizer"]
    lines.append(row(f"Otimizador:    {opt['type']}"))
    if opt["type"] == "sgd_momentum":
        lines.append(row(f"  Momentum:    {opt['momentum']}"))
        lines.append(row(f"  L2 Decay:    {opt['l2_decay']}"))
        
    lines.append(border("└", "─", "┘\n"))
    print("\n".join(lines))
