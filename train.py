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
from model.model_11 import MultiModalSurvivalModel, evaluate_model
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
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"开始新的训练运行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return logger


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs,
                device, checkpoint_dir, patience=10, gene_dropout_rate=0.4):
    """训练模型"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = logging.getLogger(__name__)

    # 存储训练历史
    history = {
        'train_loss': [], 'train_nll_loss': [], 'train_rank_loss': [],
        'train_recon_loss': [], 'train_kld_loss': [],
        'val_loss': [], 'val_nll_loss': [], 'val_rank_loss': [],
        'val_recon_loss': [], 'val_kld_loss': [],
        'val_c_index': [], 'val_tauc': []
    }

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience, mode='max', metric='c_index')
    best_c_index = 0.0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # 训练阶段
        model.train()
        train_loss_info = {
            'total_loss': 0.0, 'nll_loss': 0.0, 'rank_loss': 0.0,
            'recon_loss': 0.0, 'kld_loss': 0.0
        }

        total_samples = len(train_loader.dataset)
        num_samples_to_drop = int(total_samples * gene_dropout_rate)
        drop_indices = np.random.choice(range(total_samples), size=num_samples_to_drop, replace=False)

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_pbar):
            image = batch['image'].to(device)
            gene = batch['gene'].to(device)
            tme = batch['TME'].to(device)
            time = batch['time'].to(device)
            event = batch['event'].to(device)

            # 模拟基因缺失
            if batch_idx in drop_indices:
                gene = torch.zeros_like(gene)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(image, gene, tme, training=True)

            # 计算损失
            loss_info = model.calculate_loss(outputs, time, event)

            # 反向传播与优化
            loss_info['total_loss'].backward()
            optimizer.step()

            # 累积损失
            for key in train_loss_info:
                if key in loss_info:
                    train_loss_info[key] += loss_info[key].item() if torch.is_tensor(loss_info[key]) else loss_info[key]

            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{loss_info['total_loss'].item():.4f}",
                'nll': f"{loss_info['nll_loss'].item():.4f}",
                'rank': f"{loss_info['rank_loss'].item():.4f}"
            })

        # 计算平均训练损失
        num_batches = len(train_loader)
        for key in train_loss_info:
            train_loss_info[key] /= num_batches

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 记录训练损失
        history['train_loss'].append(train_loss_info['total_loss'])
        history['train_nll_loss'].append(train_loss_info['nll_loss'])
        history['train_rank_loss'].append(train_loss_info['rank_loss'])
        history['train_recon_loss'].append(train_loss_info['recon_loss'])
        history['train_kld_loss'].append(train_loss_info['kld_loss'])

        # 打印训练损失
        logger.info(f"Train Loss: {train_loss_info['total_loss']:.4f}, "
                    f"NLL Loss: {train_loss_info['nll_loss']:.4f}, "
                    f"Rank Loss: {train_loss_info['rank_loss']:.4f}, "
                    f"Recon Loss: {train_loss_info['recon_loss']:.4f}, "
                    f"KLD Loss: {train_loss_info['kld_loss']:.4f}")

        # 验证阶段
        logger.info("Evaluating on validation set...")
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")
        val_loss_info = evaluate_model(model, val_pbar, device)

        # 记录验证损失和性能
        history['val_loss'].append(val_loss_info['total_loss'])
        history['val_nll_loss'].append(val_loss_info['nll_loss'])
        history['val_rank_loss'].append(val_loss_info.get('rank_loss', 0.0))
        history['val_recon_loss'].append(val_loss_info.get('recon_loss', 0.0))
        history['val_kld_loss'].append(val_loss_info.get('kld_loss', 0.0))
        history['val_c_index'].append(val_loss_info['c_index'])
        history['val_tauc'].append(val_loss_info['tauc'])

        # 打印验证性能
        logger.info(f"Validation Loss: {val_loss_info['total_loss']:.4f}, "
                    f"NLL Loss: {val_loss_info['nll_loss']:.4f}, "
                    f"Rank Loss: {val_loss_info.get('rank_loss', 0.0):.4f}, "
                    f"C-index: {val_loss_info['c_index']:.4f}, "
                    f"tAUC: {val_loss_info['tauc']:.4f}")

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
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Total Loss')

    # NLL损失
    axes[0, 1].plot(history['train_nll_loss'], label='Train NLL')
    axes[0, 1].plot(history['val_nll_loss'], label='Val NLL')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('NLL Loss')
    axes[0, 1].legend()
    axes[0, 1].set_title('NLL Loss')

    # Ranking损失
    axes[0, 2].plot(history['train_rank_loss'], label='Train Rank')
    axes[0, 2].plot(history['val_rank_loss'], label='Val Rank')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Rank Loss')
    axes[0, 2].legend()
    axes[0, 2].set_title('Ranking Loss')

    # C-index
    axes[1, 0].plot(history['val_c_index'], label='Val C-index', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('C-index')
    axes[1, 0].legend()
    axes[1, 0].set_title('C-index')

    # tAUC
    axes[1, 1].plot(history['val_tauc'], label='Val tAUC', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('tAUC')
    axes[1, 1].legend()
    axes[1, 1].set_title('tAUC (IBS)')

    # VAE损失（如果有）
    if history['train_recon_loss'] and any(x > 0 for x in history['train_recon_loss']):
        axes[1, 2].plot(history['train_recon_loss'], label='Train Recon', alpha=0.7)
        axes[1, 2].plot(history['train_kld_loss'], label='Train KLD', alpha=0.7)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('VAE Loss')
        axes[1, 2].legend()
        axes[1, 2].set_title('VAE Losses')
    else:
        axes[1, 2].text(0.5, 0.5, 'No VAE Loss\n(Method: none/mean/knn)',
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('VAE Losses')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


def main():
    """主函数：模型训练和验证"""
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

    parser.add_argument("--imputation_method", type=str, default='none',
                        choices=['none', 'mean', 'knn', 'vae'],
                        help="Gene imputation method to use")
    parser.add_argument("--k_neighbors", type=int, default=5,
                        help="Number of neighbors for KNN imputation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="./result_folder",
                        help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping (number of epochs)")
    parser.add_argument("--gene_dropout_rate", type=float, default=0.4,
                        help="Ratio of samples to drop gene data during training")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Imputation method: {args.imputation_method}")

    # 创建数据集
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
        fold_checkpoint_dir = os.path.join(
            args.checkpoint_dir,
            f"{args.imputation_method}_fold_{fold_idx + 1}"
        )
        os.makedirs(fold_checkpoint_dir, exist_ok=True)

        # 设置日志
        logger = setup_logging(fold_checkpoint_dir)
        logger.info(f"Starting Fold {fold_idx + 1}/5 training with {args.imputation_method} imputation")

        # 读取split文件
        split_path = os.path.join(args.splits_folder, split_file)
        split_df = pd.read_csv(split_path)

        # 获取训练和验证sample IDs
        train_ids = split_df['train'].dropna().tolist()
        val_ids = split_df['val'].dropna().tolist()

        logger.info(f"Split file: {split_file}")
        logger.info(f"Train samples in split file: {len(train_ids)}")
        logger.info(f"Validation samples in split file: {len(val_ids)}")

        # 提取训练和验证索引
        train_indices = []
        val_indices = []

        for i in range(len(full_dataset)):
            sample = full_dataset[i]
            file_name = sample['file_name']
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
            drop_last=True
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
        model = MultiModalSurvivalModel(
            imputation_method=args.imputation_method,
            k_neighbors=args.k_neighbors
        ).to(device)

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
            patience=args.patience,
            gene_dropout_rate=args.gene_dropout_rate
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
            f.write(f"Imputation Method: {args.imputation_method}\n")
            f.write(f"Best C-index: {best_c_index:.4f} (Epoch {best_c_index_epoch})\n")
            f.write(f"Best tAUC: {best_tauc:.4f} (Epoch {best_tauc_epoch})\n")

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
    std_c_index = np.std(all_folds_results['c_index'])
    std_tauc = np.std(all_folds_results['tauc'])

    results_file = os.path.join(args.checkpoint_dir, f'{args.imputation_method}_overall_cv_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Cross-Validation Results - {args.imputation_method.upper()} Imputation\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Fold\tC-index\ttAUC\tBest Epoch\n")

        for i in range(len(all_folds_results['c_index'])):
            f.write(
                f"{i + 1}\t{all_folds_results['c_index'][i]:.4f}\t"
                f"{all_folds_results['tauc'][i]:.4f}\t{all_folds_results['best_epoch'][i]}\n"
            )

        f.write(f"{'=' * 50}\n")
        f.write(f"Average\t{avg_c_index:.4f}±{std_c_index:.4f}\t{avg_tauc:.4f}±{std_tauc:.4f}\n")

    print(f"\n{'=' * 50}")
    print(f"Cross-validation completed with {args.imputation_method.upper()} imputation!")
    print(f"Average C-index: {avg_c_index:.4f} ± {std_c_index:.4f}")
    print(f"Average tAUC: {avg_tauc:.4f} ± {std_tauc:.4f}")
    print(f"Results saved to {results_file}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
