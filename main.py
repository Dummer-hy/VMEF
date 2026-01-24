import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
# 导入自定义模块
#from model.Mutimodel_MOE_static import MultiModalSurvivalModel,evaluate_model
from model.Mutimodel_MOE_static import MultiModalSurvivalModel,evaluate_model
from model.dataloader import MultiModalDataset, collate_fn

warnings.filterwarnings("ignore")


class EarlyStopping:
    """早停机制，避免过拟合"""
    def __init__(self, patience=5, min_delta=0.0, mode='max', metric='c_index'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric = metric
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.wait_since_last_improvement = 0

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'min':
            score_improved = self.best_score - current_score > self.min_delta
        else:
            score_improved = current_score - self.best_score > self.min_delta

        if score_improved:
            self.best_score = current_score
            self.wait_since_last_improvement = 0
        else:
            self.wait_since_last_improvement += 1
            if self.wait_since_last_improvement >= self.patience:
                self.early_stop = True

        return self.early_stop


def setup_logging(checkpoint_dir):
    """设置日志 - 将日志重定向到checkpoint文件夹中的单个文件"""
    # 确保checkpoint目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置文件名 - 使用单个固定的日志文件
    log_file = os.path.join(checkpoint_dir, "training.log")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # 追加模式，保留所有日志
            logging.StreamHandler()
        ]
    )

    # 添加一个分隔符，区分不同的训练运行
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"开始新的训练运行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return logger


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs,
                device, checkpoint_dir, patience=10, gene_dropout_rate=0.4):
    """训练模型"""
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置日志
    logger = logging.getLogger(__name__)

    # 存储训练历史
    history = {
        'train_loss': [], 'train_nll_loss': [], 'train_rank_loss': [], 'train_mse_loss': [],
        'train_recon_loss': [], 'train_kld_loss': [],
        'val_loss': [], 'val_nll_loss': [], 'val_rank_loss': [], 'val_mse_loss': [],
        'val_recon_loss': [], 'val_kld_loss': [],
        'val_c_index': [], 'val_tauc': [],
        'nll_weight': [], 'rank_weight': [], 'mse_weight': [], 'recon_weight': [], 'kld_weight': [],
        'gene_present_ratio': []
    }

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience, mode='max', metric='c_index')
    best_c_index = 0.0

    for epoch in range(num_epochs):
        # 设置当前epoch，用于loss warm-up
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # 训练阶段
        model.train()
        train_loss_info = {
            'total_loss': 0.0, 'nll_loss': 0.0, 'rank_loss': 0.0,
            'mse_loss': 0.0, 'recon_loss': 0.0, 'kld_loss': 0.0
        }
        train_weight_info = {
            'nll_weight': 0.0, 'rank_weight': 0.0, 'mse_weight': 0.0,
            'recon_weight': 0.0, 'kld_weight': 0.0
        }
        gene_present_sum = 0.0

        total_samples = len(train_loader.dataset)
        num_samples_to_drop = int(total_samples * gene_dropout_rate)  # 计算丢弃基因数据的样本数
        drop_indices = np.random.choice(range(total_samples), size=num_samples_to_drop, replace=False)
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_pbar):
            image = batch['image'].to(device)
            gene = batch['gene'].to(device)
            tme = batch['TME'].to(device)
            time = batch['time'].to(device)
            event = batch['event'].to(device)
            if batch_idx in drop_indices:
                gene = torch.zeros_like(gene)  # 将这些样本的基因数据置为零
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(image, gene, tme)

            # 计算损失
            loss_info = model.calculate_loss(outputs, time, event)

            # 反向传播与优化
            loss_info['total_loss'].backward()
            optimizer.step()

            # 累积损失和权重
            for key in train_loss_info:
                if key in loss_info:
                    train_loss_info[key] += loss_info[key].item() if torch.is_tensor(loss_info[key]) else loss_info[key]

            for key in train_weight_info:
                if key in loss_info:
                    train_weight_info[key] += loss_info[key]

            # 记录gene模态存在比例
            if 'gene_present_ratio' in loss_info:
                gene_present_sum += loss_info['gene_present_ratio']

            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{loss_info['total_loss'].item():.4f}",
                'nll': f"{loss_info['nll_loss'].item():.4f}",
                'mse': f"{loss_info['mse_loss'].item():.4f}",
                'gene%': f"{loss_info['gene_present_ratio']:.2f}"
            })

        # 计算平均训练损失和权重
        num_batches = len(train_loader)
        for key in train_loss_info:
            train_loss_info[key] /= num_batches

        for key in train_weight_info:
            train_weight_info[key] /= num_batches

        gene_present_ratio = gene_present_sum / num_batches

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 记录训练损失和权重
        history['train_loss'].append(train_loss_info['total_loss'])
        history['train_nll_loss'].append(train_loss_info['nll_loss'])
        history['train_rank_loss'].append(train_loss_info['rank_loss'])
        history['train_mse_loss'].append(train_loss_info['mse_loss'])
        history['train_recon_loss'].append(train_loss_info['recon_loss'])
        history['train_kld_loss'].append(train_loss_info['kld_loss'])

        history['nll_weight'].append(train_weight_info['nll_weight'])
        history['rank_weight'].append(train_weight_info['rank_weight'])
        history['mse_weight'].append(train_weight_info['mse_weight'])
        history['recon_weight'].append(train_weight_info['recon_weight'])
        history['kld_weight'].append(train_weight_info['kld_weight'])

        history['gene_present_ratio'].append(gene_present_ratio)

        # 打印训练损失
        logger.info(f"Train Loss: {train_loss_info['total_loss']:.4f}, "
                    f"NLL Loss: {train_loss_info['nll_loss']:.4f}, "
                    f"Rank Loss: {train_loss_info['rank_loss']:.4f}, "
                    f"MSE Loss: {train_loss_info['mse_loss']:.4f}, "
                    f"Recon Loss: {train_loss_info['recon_loss']:.4f}, "
                    f"KLD Loss: {train_loss_info['kld_loss']:.4f}, "
                    f"Gene Present: {gene_present_ratio:.4f}")

        # 验证阶段
        logger.info("Evaluating on validation set...")

        # 使用tqdm创建验证进度条
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")
        val_loss_info = evaluate_model(model, val_pbar, device)

        # 记录验证损失和性能
        history['val_loss'].append(val_loss_info['total_loss'])
        history['val_nll_loss'].append(val_loss_info['nll_loss'])
        history['val_rank_loss'].append(val_loss_info.get('rank_loss', 0.0))
        history['val_mse_loss'].append(val_loss_info['mse_loss'])
        history['val_recon_loss'].append(val_loss_info.get('recon_loss', 0.0))
        history['val_kld_loss'].append(val_loss_info.get('kld_loss', 0.0))

        history['val_c_index'].append(val_loss_info['c_index'])
        history['val_tauc'].append(val_loss_info['tauc'])

        # 打印验证性能
        logger.info(f"Validation Loss: {val_loss_info['total_loss']:.4f}, "
                    f"NLL Loss: {val_loss_info['nll_loss']:.4f}, "
                    f"MSE Loss: {val_loss_info['mse_loss']:.4f}, "
                    f"C-index: {val_loss_info['c_index']:.4f}, "
                    f"tAUC: {val_loss_info['tauc']:.4f},")

        # 保存当前epoch的模型
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'c_index': val_loss_info['c_index'],
            'tauc': val_loss_info['tauc'],
            'loss': val_loss_info['total_loss'],
        }, checkpoint_path)

        # 保存最佳模型
        if val_loss_info['c_index'] > best_c_index:
            best_c_index = val_loss_info['c_index']
            logger.info(f"New best model with C-index: {best_c_index:.4f}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'c_index': best_c_index,
                'tauc': val_loss_info['tauc'],
                'loss': val_loss_info['total_loss'],
            }, os.path.join(checkpoint_dir, 'best_model.pth'))

        # 检查是否早停
        if early_stopping(val_loss_info['c_index']):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # 每个epoch结束后保存历史记录
        pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, 'training_history.csv'), index=False)

        # 每5个epoch绘制并保存训练历史曲线
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1 or early_stopping.early_stop:
            plot_training_history(history, checkpoint_dir)

    # 确保最终历史记录和图形已保存
    pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, 'final_training_history.csv'), index=False)

    return model, history

def plot_training_history(history, save_dir):
    """绘制训练历史曲线"""
def main():
    """主函数：模型训练和验证"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train and validate multimodal survival model")
    parser.add_argument("--csv_path", type=str,
                         default=r'D:\Documents\Pycharm\MIL\PORPOISE\datasets\TCGA-SKCM\TCGA-SKCM.csv',
                         help="Path to the CSV file containing sample information")
    parser.add_argument("--wsi_folder", type=str,
                         default=r'D:\Documents\Pycharm\MIL\PORPOISE\datasets\data_root\skcm\resnet50',
                         help="Path to the folder containing WSI image files (.pt)")
    parser.add_argument("--splits_folder", type=str,
                         default=r"D:\Documents\Pycharm\MIL\PORPOISE\splits\skcm",
                         help="Path to the folder containing 5-fold CV split CSV files")

    # parser.add_argument("--csv_path", type=str,
    #                    default=r'D:\Documents\Pycharm\MIL\PORPOISE\datasets\TCGA-LUSC\TCGA-LUSC.csv',
    #                    help="Path to the CSV file containing sample information")
    # parser.add_argument("--wsi_folder", type=str,
    #                    default=r'D:\Documents\Pycharm\MIL\PORPOISE\datasets\data_root\lusc\resnet50',
    #                    help="Path to the folder containing WSI image files (.pt)")
    # parser.add_argument("--splits_folder", type=str,
    #                    default=r"D:\Documents\Pycharm\MIL\PORPOISE\splits\lusc",
    #                    help="Path to the folder containing 5-fold CV split CSV files")
    # parser.add_argument("--csv_path", type=str,
    #                     default=r'D:\Documents\Pycharm\MIL\PORPOISE\datasets\TCGA-HNSC\TCGA-HNSC.csv',
    #                     help="Path to the CSV file containing sample information")
    # parser.add_argument("--wsi_folder", type=str,
    #                     default=r'D:\Documents\Pycharm\MIL\PORPOISE\datasets\data_root\hnsc\resnet50',
    #                     help="Path to the folder containing WSI image files (.pt)")
    # parser.add_argument("--splits_folder", type=str,
    #                     default=r"D:\Documents\Pycharm\MIL\PORPOISE\splits\hnsc",
    #                     help="Path to the folder containing 5-fold CV split CSV files")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="./result_folder",
                        help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping (number of epochs)")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据集 - 不进行任何补丁操作
    full_dataset = MultiModalDataset(csv_path=args.csv_path, WSI_folder=args.wsi_folder, aggregate_patches=True)

    # 获取splits文件夹中的所有CSV文件并排序
    split_files = sorted([f for f in os.listdir(args.splits_folder) if f.endswith('.csv')])

    # 存储所有折叠的结果
    all_folds_results = {
        'c_index': [],
        'tauc': [],
        'best_epoch': []
    }

    # 对每个fold进行训练和验证
    for fold_idx, split_file in enumerate(split_files):
        print(f"\n{'=' * 50}")
        print(f"Starting training for Fold {fold_idx + 1}/5")
        print(f"{'=' * 50}")

        # 为当前折叠创建检查点目录
        fold_checkpoint_dir = os.path.join(args.checkpoint_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_checkpoint_dir, exist_ok=True)

        # 设置日志 - 将日志写入当前折叠的checkpoint文件夹
        logger = setup_logging(fold_checkpoint_dir)
        logger.info(f"Starting Fold {fold_idx + 1}/5 training")

        # 读取split文件
        split_path = os.path.join(args.splits_folder, split_file)
        split_df = pd.read_csv(split_path)

        # 获取训练和验证sample IDs
        # 转换为列表，过滤空值
        train_ids = split_df['train'].dropna().tolist()
        val_ids = split_df['val'].dropna().tolist()

        logger.info(f"Split file: {split_file}")
        logger.info(f"Train samples in split file: {len(train_ids)}")
        logger.info(f"Validation samples in split file: {len(val_ids)}")

        # 提取训练和验证索引
        train_indices = []
        val_indices = []

        # 打印一些示例文件名以便调试
        for i in range(len(full_dataset)):
            sample = full_dataset[i]
            file_name = sample['file_name']

            # 从文件名中提取TCGA ID - 取前12个字符 (TCGA-XX-XXXX 格式)
            tcga_id = file_name[:12]

            if tcga_id in train_ids:
                train_indices.append(i)
            elif tcga_id in val_ids:
                val_indices.append(i)

        logger.info(f"Found {len(train_indices)} training and {len(val_indices)} validation samples in dataset")

        # 如果没有找到足够的样本，跳过这个fold
        if len(train_indices) == 0 or len(val_indices) == 0:
            logger.warning(f"Insufficient samples found for fold {fold_idx + 1}. Skipping this fold.")
            continue

        # 创建数据加载器
        train_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True  # 删除最后不满一个批次的样本，避免单样本批次问题
        )

        val_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        # 创建模型
        model = MultiModalSurvivalModel().to(device)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # 创建优化器和学习率调度器
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 训练模型
        logger.info("Starting model training...")
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            device=device,
            checkpoint_dir=fold_checkpoint_dir,
            patience=args.patience
        )

        # 记录最终验证性能
        best_c_index = max(history['val_c_index'])
        best_c_index_epoch = history['val_c_index'].index(best_c_index) + 1
        best_tauc = max(history['val_tauc'])
        best_tauc_epoch = history['val_tauc'].index(best_tauc) + 1

        logger.info("Training completed. Final validation performance:")
        logger.info(f"Best C-index: {best_c_index:.4f} (Epoch {best_c_index_epoch})")
        logger.info(f"Best tAUC: {best_tauc:.4f} (Epoch {best_tauc_epoch})")

        # 保存最终的验证性能结果
        with open(os.path.join(fold_checkpoint_dir, 'validation_results.txt'), 'w') as f:
            f.write(f"Best C-index: {best_c_index:.4f} (Epoch {best_c_index_epoch})\n")
            f.write(f"Best tAUC: {best_tauc:.4f} (Epoch {best_tauc_epoch})\n")
            f.write(f"Final NLL Weight: {history['nll_weight'][-1]:.4f}\n")
            f.write(f"Final MSE Weight: {history['mse_weight'][-1]:.4f}\n")

        # 存储当前折叠的结果
        all_folds_results['c_index'].append(best_c_index)
        all_folds_results['tauc'].append(best_tauc)
        all_folds_results['best_epoch'].append(best_c_index_epoch)

    # 检查是否完成了任何折叠的训练
    if len(all_folds_results['c_index']) == 0:
        print("No folds were successfully trained. Please check your dataset and split files.")
        return

    # 计算并保存所有折叠的平均性能
    avg_c_index = np.mean(all_folds_results['c_index'])
    avg_tauc = np.mean(all_folds_results['tauc'])

    with open(os.path.join(args.checkpoint_dir, 'overall_cv_results.txt'), 'w') as f:
        f.write(f"Cross-Validation Results\n")
        f.write(f"{'=' * 30}\n")
        f.write(f"Fold\tC-index\ttAUC\tBest Epoch\n")

        for i in range(len(all_folds_results['c_index'])):
            f.write(
                f"{i + 1}\t{all_folds_results['c_index'][i]:.4f}\t{all_folds_results['tauc'][i]:.4f}\t{all_folds_results['best_epoch'][i]}\n")

        f.write(f"{'=' * 30}\n")
        f.write(f"Average\t{avg_c_index:.4f}\t{avg_tauc:.4f}\n")

    print(f"\n{'=' * 50}")
    print(f"Cross-validation completed!")
    print(f"Average C-index across {len(all_folds_results['c_index'])} folds: {avg_c_index:.4f}")
    print(f"Average tAUC across {len(all_folds_results['c_index'])} folds: {avg_tauc:.4f}")
    print(f"Results saved to {args.checkpoint_dir}/overall_cv_results.txt")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
