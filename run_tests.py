#!/usr/bin/env python3
"""
Script para execução automatizada de experimentos do MLP.
Varre o diretório 'configs/' procurando por arquivos 'test_*.json',
executa cada um via 'main.py' e chama 'gerar_grafico.py' para gerar as matrizes de confusão.
"""

import os
import sys
import glob
import json
import subprocess

def run_experiment(config_path):
    print("\n" + "="*80)
    print(f" Iniciando Experimento para: {config_path}")
    print("="*80)
    
    # Carrega a configuração do JSON para pegar os dados necessários
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"[Erro] Falha ao ler {config_path}: {e}")
        return False

    exp_id = config.get("experiment_id")
    use_cv = config.get("use_cross_validation", False)
    
    if not exp_id:
        print(f"[Erro] 'experiment_id' não especificado em {config_path}")
        return False
        
    if not exp_id.startswith("test_"):
        print(f"[Aviso] O experiment_id '{exp_id}' deve iniciar com 'test_'. Corrigindo para test_{exp_id}...")
        exp_id = f"test_{exp_id}"
        config["experiment_id"] = exp_id
        # Grava de volta com o id corrigido para consistência
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"[Erro] Falha ao reescrever {config_path}: {e}")

    # Executa o main.py passando a config
    print(f"[Executando] python main.py {config_path}")
    cmd_main = [sys.executable, "main.py", config_path]
    result_main = subprocess.run(cmd_main, capture_output=False)
    
    if result_main.returncode != 0:
        print(f"[Erro] main.py falhou ao executar a config {config_path} com código de retorno {result_main.returncode}")
        return False
        
    # Determina o caminho esperado do relatório JSON com base no tipo de validação
    if use_cv:
        report_filename = f"{exp_id}_cross_validation_report.json"
    else:
        report_filename = f"{exp_id}_report.json"
        
    report_path = os.path.join("outputs", exp_id, "reports", report_filename)
    
    if not os.path.exists(report_path):
        print(f"[Erro] Arquivo de relatório não foi encontrado no caminho esperado: {report_path}")
        return False
        
    # Executa gerar_grafico.py passando o caminho do relatório
    print(f"[Gerando Gráficos] python gerar_grafico.py {report_path}")
    cmd_plot = [sys.executable, "gerar_grafico.py", report_path]
    result_plot = subprocess.run(cmd_plot, capture_output=False)
    
    if result_plot.returncode != 0:
        print(f"[Erro] gerar_grafico.py falhou ao processar {report_path}")
        return False
        
    print(f"[Sucesso] Experimento {exp_id} concluído e gráficos salvos!")
    return True

def main():
    configs_dir = "configs"
    if not os.path.exists(configs_dir):
        print(f"[Erro] Diretório {configs_dir}/ não encontrado.")
        sys.exit(1)
        
    # Busca arquivos test_*.json no diretório configs/
    search_pattern = os.path.join(configs_dir, "test_*.json")
    config_files = glob.glob(search_pattern)
    
    if not config_files:
        print(f"[Erro] Nenhum arquivo correspondente a 'test_*.json' foi encontrado em '{configs_dir}/'.")
        sys.exit(1)
        
    print(f"Encontrados {len(config_files)} experimentos para rodar: {[os.path.basename(c) for c in config_files]}")
    
    success_count = 0
    for config_path in sorted(config_files):
        if run_experiment(config_path):
            success_count += 1
            
    print("\n" + "="*80)
    print(f" EXECUÇÃO FINALIZADA. Sucessos: {success_count}/{len(config_files)}")
    print("="*80)

if __name__ == "__main__":
    main()
