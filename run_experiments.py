#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np
import torch
from models.predicate_config import Config
from validate_model import validate_baseline, validate_generalization, test_robustness
from ablation_experiments import (
    ablation_bert_freeze,
    ablation_classifier_structure,
    ablation_input_length,
    ablation_pretrained_models
)
from visualize_results import save_experiment_results, visualize_experiment_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行模型验证和消融实验')
    parser.add_argument('--all', action='store_true', help='运行所有实验')
    parser.add_argument('--baseline', action='store_true', help='运行基准性能评估')
    parser.add_argument('--generalization', action='store_true', help='运行泛化能力测试')
    parser.add_argument('--robustness', action='store_true', help='运行鲁棒性测试')
    parser.add_argument('--bert-freeze', action='store_true', help='运行BERT冻结消融实验')
    parser.add_argument('--classifier', action='store_true', help='运行分类器结构消融实验')
    parser.add_argument('--input-length', action='store_true', help='运行输入长度消融实验')
    parser.add_argument('--pretrained', action='store_true', help='运行预训练模型比较实验')
    parser.add_argument('--external-data', type=str, default='', help='外部数据集路径')
    parser.add_argument('--visualize-only', action='store_true', help='只可视化已有结果')

    # 鲁棒性测试：python
    # run_experiments.py - -robustness
    # BERT冻结消融实验：python
    # run_experiments.py - -bert - freeze
    # 分类器结构消融实验：python
    # run_experiments.py - -classifier
    # 输入长度消融实验：python
    # run_experiments.py - -input - length
    # 或者一次性运行所有实验：python
    # run_experiments.py - -all
    return parser.parse_args()

def setup_environment():
    """设置实验环境"""
    # 设置工作目录为项目根目录
    this_file = os.path.abspath(__file__)
    project_root = os.path.dirname(this_file)
    os.chdir(project_root)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 设置随机种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，使用CPU")

def collect_results():
    """收集实验结果"""
    results = {}
    
    # 尝试加载已有结果
    result_file = 'results/experiment_results.json'
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            print(f"警告: 无法加载已有结果文件 {result_file}")
    
    # 添加类别名称
    config = Config()
    results['class_names'] = config.class_list
    
    return results

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置环境
    setup_environment()
    
    # 如果只是可视化已有结果
    if args.visualize_only:
        print("只可视化已有结果...")
        visualize_experiment_results()
        return
    
    # 收集实验结果
    results = collect_results()
    
    # 运行实验
    if args.all or args.baseline:
        print("\n===== 运行基准性能评估 =====")
        validate_baseline()
    
    if args.all or args.generalization:
        if args.external_data:
            print(f"\n===== 运行泛化能力测试: {args.external_data} =====")
            validate_generalization(args.external_data)
        else:
            print("警告: 未指定外部数据集路径，跳过泛化能力测试")
    
    if args.all or args.robustness:
        print("\n===== 运行鲁棒性测试 =====")
        test_robustness()
    
    if args.all or args.bert_freeze:
        print("\n===== 运行BERT冻结消融实验 =====")
        ablation_bert_freeze()
    
    if args.all or args.classifier:
        print("\n===== 运行分类器结构消融实验 =====")
        ablation_classifier_structure()
    
    if args.all or args.input_length:
        print("\n===== 运行输入长度消融实验 =====")
        ablation_input_length()
    
    if args.all or args.pretrained:
        print("\n===== 运行预训练模型比较实验 =====")
        ablation_pretrained_models()
    
    # 保存和可视化结果
    save_experiment_results(results)
    visualize_experiment_results()

if __name__ == "__main__":
    main() 