"""
Atividade de IA. Integrantes:

Hugo Cardoso Ferreira de Araújo - 15459500
Higor Gabriel de Freitas - 15575879
Enrico Lechar de Barros Aranha - 15449652
Renan Rodrigues Moreira - 15744874
Clara Pires Campardo - 15446433
"""

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
