from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import logging
import argparse
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    def __init__(self, csv_path, WSI_folder, transform=None, gene_cols=None, tme_cols=None, aggregate_patches=True):
        """
        Args:
            csv_path (str): CSV文件路径，包含样本信息
            WSI_folder (str): 图像数据文件夹路径
            transform (callable, optional): 数据预处理操作（如标准化、裁剪等）
            gene_cols (list, optional): 基因数据列的索引或名称，默认为[15:180]
            tme_cols (list, optional): TME数据列的索引或名称，默认为最后180列
            aggregate_patches (bool, optional): 是否对patch特征求平均，默认为True
        """
        self.csv_path = csv_path
        self.WSI_folder = WSI_folder
        self.transform = transform
        self.aggregate_patches = aggregate_patches

        # 加载数据
        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f'Successfully loaded data from {csv_path}, found {len(self.df)} samples')
        except Exception as e:
            logger.error(f'Failed to load data from {csv_path}: {e}')
            raise
        # self.gene_cols = list(range(15, 180)) if gene_cols is None else gene_cols
        # self.tme_cols = list(range(-180, 0)) if tme_cols is None else tme_cols
        total_cols = 3363 + 180  # 3363 gene 列 + 180 tme 列
        self.tme_cols = list(range(total_cols - 180, total_cols)) if tme_cols is None else tme_cols
        self.gene_cols = list(range(total_cols - 180 - 3363, total_cols - 180)) if gene_cols is None else gene_cols
        self.df['image_file'] = self.df['File Name'].apply(lambda x: x.replace('.svs', '.pt'))
        self.df['stage_label'] = self.df['Stage_label']
        self.df['event'] = self.df['event']
        self.df['time'] = self.df['time']
        self.process_molecular_data()
        self.clean_data()

    def process_molecular_data(self):
        """处理并验证分子数据（基因和TME）"""
        try:
            # 定义检查和转换函数
            def check_and_convert(row):
                try:
                    row = pd.to_numeric(row, errors='coerce')
                    # 检测异常值并替换（例如，用3倍标准差作为阈值）
                    mean, std = row.mean(), row.std()
                    threshold = mean + 3 * std
                    row = row.mask(row > threshold, threshold)
                    return row.fillna(0).values
                except Exception:
                    logger.warning(f"Error converting row to numeric, using zeros instead")
                    return np.zeros(len(row))

            # 应用检查函数
            self.df['gene_data'] = self.df.iloc[:, self.gene_cols].apply(check_and_convert, axis=1)
            self.df['TME_data'] = self.df.iloc[:, self.tme_cols].apply(check_and_convert, axis=1)
            logger.info("Successfully processed molecular data")
        except Exception as e:
            logger.error(f"Error in processing molecular data: {e}")
            raise

    def clean_data(self):
        """清洗数据：删除标签为空或缺失模态的样本"""
        original_len = len(self.df)

        # 删除标签为空的样本
        self.df = self.df.dropna(subset=['stage_label'])
        logger.info(f'Removed {original_len - len(self.df)} samples with missing labels')

        # 检查图像文件是否存在
        missing_images = []
        for idx, row in self.df.iterrows():
            image_path = os.path.join(self.WSI_folder, row['image_file'])
            if not os.path.exists(image_path):
                missing_images.append(idx)

        if missing_images:
            logger.warning(f'Found {len(missing_images)} samples with missing image files')
            self.df = self.df.drop(missing_images)

        # 删除模态不全的样本
        valid_samples = (
                self.df['gene_data'].apply(lambda x: len(x) > 0) &  # 确保基因数据不为空
                self.df['TME_data'].apply(lambda x: len(x) > 0)  # 确保TME数据不为空
        )
        self.df = self.df[valid_samples]

        logger.info(f'Final cleaned dataset size: {len(self.df)} samples')

    def __len__(self):
        """返回数据集的大小"""
        return len(self.df)

    def __getitem__(self, idx):
        """返回每个样本的图像特征、基因数据、TME数据、标签和生存信息"""
        sample = self.df.iloc[idx]

        # 读取图像数据（.pt文件）
        try:
            image_file = sample['image_file']
            image_path = os.path.join(self.WSI_folder, image_file)
            image_data = torch.load(image_path)

            # 根据参数选择是否聚合patch特征
            if self.aggregate_patches:
                # 聚合图像特征 - 求平均得到slide表示
                image_features = self.aggregate_patch_features(image_data)
            else:
                # 不聚合，直接使用原始patch特征
                image_features = image_data

            # 应用变换（如果有）
            if self.transform:
                image_features = self.transform(image_features)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # 根据选择的聚合模式创建相应形状的全零张量
            if self.aggregate_patches:
                image_features = torch.zeros(1, 1024, dtype=torch.float32)
            else:
                image_features = torch.zeros(10, 1024, dtype=torch.float32)

        # 获取基因数据和TME数据
        gene_data = torch.tensor(sample['gene_data'], dtype=torch.float32)
        TME_data = torch.tensor(sample['TME_data'], dtype=torch.float32)

        # 获取标签和生存信息
        label = torch.tensor(sample['stage_label'], dtype=torch.long)
        time = torch.tensor(sample['time'], dtype=torch.float32)
        event = torch.tensor(sample['event'], dtype=torch.float32)
        return {
            'image': image_features,
            'gene': gene_data,
            'TME': TME_data,
            'label': label,
            'time': time,
            'event': event,
            'file_name': sample['image_file']
        }

    def aggregate_patch_features(self, patches):
        """
        聚合patch特征以获得slide级别的表示

        Args:
            patches (Tensor): 形状为 [patch_num, 1024] 的patch特征

        Returns:
            Tensor: 形状为 [1, 1024] 的聚合特征
        """
        # 确保输入是张量
        if not isinstance(patches, torch.Tensor):
            patches = torch.tensor(patches, dtype=torch.float32)

        # 检查张量维度
        if patches.dim() != 2:
            logger.warning(f"Expected 2D tensor, got {patches.dim()}D instead. Reshaping...")
            patches = patches.view(-1, 1024)

        # 计算平均值得到slide表示
        slide_representation = torch.mean(patches, dim=0, keepdim=True)

        return slide_representation


def collate_fn(batch):
    """自定义 collate_fn 函数，处理批量数据，支持多示例学习和聚合模式"""

    # 提取每个样本中的数据
    images = [item['image'] for item in batch]
    genes = [item['gene'] for item in batch]
    TMEs = [item['TME'] for item in batch]
    labels = [item['label'] for item in batch]
    times = [item['time'] for item in batch]
    events = [item['event'] for item in batch]
    file_names = [item['file_name'] for item in batch]

    # 检查是否为聚合模式（通过检查第一个样本的形状）
    is_aggregated = images[0].dim() == 2 and images[0].size(0) == 1

    if is_aggregated:
        # 聚合模式: 处理已经聚合的图像数据
        images = torch.stack(images, dim=0)  # 形状: [batch_size, 1, 1024]
        images = images.squeeze(1)  # 形状: [batch_size, 1024]
        patch_counts = None  # 聚合模式下不需要patch计数
    else:
        # 非聚合模式: 处理多示例学习的patch数据
        patch_counts = [img.shape[0] for img in images]  # 记录每个样本的patch数量
        # images保持为列表，每个元素形状为 [patch_num, 1024]

    # 堆叠其他数据
    genes = torch.stack(genes, dim=0)  # 形状: [batch_size, gene_dim]
    TMEs = torch.stack(TMEs, dim=0)  # 形状: [batch_size, tme_dim]
    labels = torch.stack(labels, dim=0)  # 形状: [batch_size]
    times = torch.stack(times, dim=0)  # 形状: [batch_size]
    events = torch.stack(events, dim=0)  # 形状: [batch_size]

    # 返回结果，包括聚合模式标志
    result = {
        'image': images,
        'gene': genes,
        'TME': TMEs,
        'label': labels,
        'time': times,
        'event': events,
        'file_name': file_names,
        'is_aggregated': is_aggregated  # 添加标志以便模型知道数据类型
    }

    # 如果是非聚合模式，添加patch计数
    if not is_aggregated:
        result['patch_counts'] = patch_counts

    return result



