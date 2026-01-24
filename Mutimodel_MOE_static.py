import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


def init_weights(m):
    """初始化网络权重"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def R_set(x):
    '''创建风险集指示矩阵，其中 T_j >= T_i。
    注意输入数据应该按照降序排列。
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return indicator_matrix


class MLP(nn.Module):
    """多层感知机模块，用于单模态特征提取"""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3, use_residual=True):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        self.use_residual = use_residual and input_dim == output_dim

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*layers)
        # 初始化权重
        self.apply(init_weights)

    def forward(self, x):
        out = self.layers(x)
        if self.use_residual:
            out = out + x  # 添加残差连接
        return out

class CrossAttention(nn.Module):
    """交叉注意力模块，用于两个模态之间的交互"""

    def __init__(self, dim, heads=4, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(dim, dim)

        # 初始化权重
        self.apply(init_weights)

    def forward(self, x, y):
        # x: query source, y: key/value source
        batch_size = x.size(0)

        # 生成查询、键和值
        q = self.q(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(y).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(y).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        # 输出投影
        output = self.out(context)

        return output.squeeze(1)  # 返回 [batch_size, dim]
class SelfAttention(nn.Module):
    """自注意力模块，用于增强单模态特征表示"""

    def __init__(self, dim, heads=4, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(dim, dim)

        # 初始化权重
        self.apply(init_weights)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.q(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        output = self.out(context)
        return output.squeeze(1)

class GatedAttention(nn.Module):
    """门控注意力模块，用于融合特征"""

    def __init__(self, dim, dropout_rate=0.1):
        super(GatedAttention, self).__init__()
        self.gate = nn.Linear(2 * dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(2 * dim, dim)
        self.layer_norm = nn.LayerNorm(dim)  # 添加层归一化

        # 初始化权重
        self.apply(init_weights)

    def forward(self, x, y):
        # x, y: 两个不同模态的特征 [batch_size, dim]

        # 连接两个特征
        combined = torch.cat([x, y], dim=1)  # [batch_size, 2*dim]

        # 计算门控值
        gate_value = torch.sigmoid(self.gate(combined))  # [batch_size, 1]

        # 应用门控
        gated_combined = gate_value * combined
        gated_combined = self.dropout(gated_combined)

        # 投影为最终融合特征
        fusion = self.projection(gated_combined)  # [batch_size, dim]
        fusion = self.layer_norm(fusion)  # 应用层归一化

        return fusion, gate_value


class DynamicWeightedLoss(nn.Module):
    """动态加权损失函数"""

    def __init__(self, num_losses=3, init_weights=None):
        super(DynamicWeightedLoss, self).__init__()

        if init_weights is None:
            # 默认初始权重
            init_weights = [0.6, 0.3, 0.1]  # NLL, Ranking, MSE

        assert len(init_weights) == num_losses, "初始权重数量必须等于损失函数数量"
        init_weights = [w / sum(init_weights) for w in init_weights]

        # 使用softmax参数化权重
        self.log_weights = nn.Parameter(torch.tensor([np.log(w) for w in init_weights], dtype=torch.float))

    def forward(self, losses):
        """
        根据动态权重计算总损失

        参数:
            losses: 包含多个损失的列表
        """
        weights = F.softmax(self.log_weights, dim=0)
        total_loss = sum(w * l for w, l in zip(weights, losses))
        return total_loss, weights.detach().cpu().numpy()


class NegativeLogLikelihood(nn.Module):
    """负对数似然损失函数（用于生存分析）"""

    def __init__(self, reduction='mean'):
        super(NegativeLogLikelihood, self).__init__()
        self.reduction = reduction

    def forward(self, pred, time, event):
        order = torch.argsort(time, descending=True)
        risk_pred = pred[order]
        event = event[order]
        n_observed = event.sum()
        if n_observed == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        ytime_indicator = R_set(time[order]).to(pred.device)
        risk_set_sum = torch.log(torch.sum(ytime_indicator * torch.exp(risk_pred.view(-1, 1)), dim=1) + 1e-8)
        neg_likelihood = -torch.sum((risk_pred - risk_set_sum) * event)
        if self.reduction == 'mean':
            neg_likelihood = neg_likelihood / n_observed

        return neg_likelihood

class RankingLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super(RankingLoss, self).__init__()
        self.sigma = sigma  # 控制sigmoid函数的陡度

    def forward(self, pred, time, event):
        """
        参数:
            pred: 预测的风险分数 [batch_size]
            time: 生存时间 [batch_size]
            event: 事件状态 [batch_size]
        """
        # 如果没有观察到事件，返回零损失
        if event.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        R = pred.reshape(-1)
        n_samples = R.size(0)
        event = event.bool()
        observed_idx = torch.where(event)[0]
        n_observed = observed_idx.size(0)
        if n_observed == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        total_loss = 0
        total_comparisons = 0

        for i in observed_idx:
            for j in range(n_samples):
                if i == j:
                    continue  # 跳过自身比较
                if time[i] < time[j]:
                    pair_prob = torch.sigmoid((R[i] - R[j]) / self.sigma)
                    total_loss += -torch.log(pair_prob + 1e-8)
                    total_comparisons += 1
        if total_comparisons == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        return total_loss / total_comparisons


class SNN(nn.Module):
    """Self-Normalizing Neural Network (SNN) 实现，增加稳定性措施"""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.05):
        super(SNN, self).__init__()

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            # 使用特殊的权重初始化
            linear = nn.Linear(dims[i], dims[i + 1])
            # 使用合适的初始化方法
            nn.init.xavier_normal_(linear.weight, gain=nn.init.calculate_gain('selu'))
            nn.init.zeros_(linear.bias)

            layers.append(linear)

            if i < len(dims) - 2:  # 不在最后一层添加激活函数和dropout
                # SELU激活函数是SNN的核心
                layers.append(nn.SELU())
                # Alpha Dropout保持自归一化性质
                layers.append(nn.AlphaDropout(p=dropout_rate))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 添加输入裁剪，防止极端值
        x = torch.clamp(x, -10, 10)
        return self.network(x)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts模块，增加数值稳定性"""

    def __init__(self, feature_dim, num_experts=2):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        # 门控网络，用于计算每个专家的权重
        self.gating_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout
            nn.Linear(feature_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, mu_list, logvar_list):
        """MoE融合多个分布"""
        # 将多个专家的参数拼接
        mu_stack = torch.stack(mu_list, dim=1)  # [batch_size, num_experts, feature_dim]
        logvar_stack = torch.stack(logvar_list, dim=1)  # [batch_size, num_experts, feature_dim]

        # 裁剪logvar防止数值不稳定
        logvar_stack = torch.clamp(logvar_stack, -10, 10)

        # 计算门控输入（使用所有专家的特征）
        gating_input = torch.cat([mu_list[0], mu_list[1]], dim=1)

        # 获取每个专家的权重
        expert_weights = self.gating_network(gating_input)  # [batch_size, num_experts]

        # 添加小的epsilon防止除零
        expert_weights = expert_weights + 1e-8
        expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

        # 扩展权重维度以匹配特征维度
        expert_weights = expert_weights.unsqueeze(2)  # [batch_size, num_experts, 1]

        # 加权融合均值
        mu_combined = torch.sum(mu_stack * expert_weights, dim=1)  # [batch_size, feature_dim]

        # 简化方差融合，避免数值不稳定
        var_stack = torch.exp(torch.clamp(logvar_stack, -10, 10))
        var_combined = torch.sum(var_stack * expert_weights, dim=1)
        logvar_combined = torch.log(var_combined + 1e-8)

        # 再次裁剪输出
        mu_combined = torch.clamp(mu_combined, -10, 10)
        logvar_combined = torch.clamp(logvar_combined, -10, 10)

        return mu_combined, logvar_combined


class MultiModalSurvivalModel(nn.Module):
    """多模态生存分析模型，增加数值稳定性措施"""

    def __init__(self, image_dim=1024, gene_dim=3363, tme_dim=180, feature_dim=64, hidden_dims=[256, 128],
                 gene_missing_threshold=100000):
        super(MultiModalSurvivalModel, self).__init__()

        # 模态特定的特征提取网络
        self.image_mlp = MLP(image_dim, hidden_dims, feature_dim, use_residual=False)
        self.feature_dim = feature_dim
        # 使用SNN处理基因特征
        self.gene_snn = SNN(
            input_dim=gene_dim,
            hidden_dims=hidden_dims,
            output_dim=feature_dim,
            dropout_rate=0.05
        )

        self.tme_mlp = MLP(tme_dim, hidden_dims, feature_dim, use_residual=False)

        # 添加自注意力增强单模态特征表示
        self.image_self_attn = SelfAttention(feature_dim)
        self.gene_self_attn = SelfAttention(feature_dim)
        self.tme_self_attn = SelfAttention(feature_dim)

        # 交叉注意力模块（基因和TME之间）
        self.cross_attention_gene2tme = CrossAttention(feature_dim)
        self.cross_attention_tme2gene = CrossAttention(feature_dim)

        # 门控注意力模块（图像和融合特征之间）
        self.gated_attention = GatedAttention(feature_dim)

        # 添加图像特征变换MLP，用于MSE损失计算前的特征对齐
        self.image_transform_mlp = MLP(feature_dim, [feature_dim * 2], feature_dim, dropout_rate=0.2, use_residual=True)

        # VAE编码器 - 一个用于基因特征，一个用于TME特征
        self.encoder_gene = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim * 2)  # 输出mu和logvar
        )

        self.encoder_tme = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim * 2)  # 输出mu和logvar
        )

        # VAE解码器 - 共享解码器，用于生成融合特征
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # MoE模块替代PoE
        self.mixture_of_experts = MixtureOfExperts(feature_dim, num_experts=2)

        # 生存预测头
        self.survival_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # 初始化生存预测头权重
        self.survival_head.apply(self.init_weights)

        # 损失函数
        self.nll_loss = NegativeLogLikelihood(reduction='mean')
        self.ranking_loss = RankingLoss(sigma=0.1)
        self.mse_loss = nn.MSELoss()

        # 使用动态权重组合
        self.dynamic_weight = DynamicWeightedLoss(num_losses=3, init_weights=[0.7, 0.3, 0.1])

        # 添加gate值正则化的alpha参数
        self.gate_reg_alpha = 0.01

        # 模态缺失阈值
        self.gene_missing_threshold = gene_missing_threshold

    def init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def reparameterize(self, mu, logvar):
        """VAE重参数化采样，增加数值稳定性"""
        # 裁剪logvar防止数值爆炸
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, gene, tme):
        """前向传播，增加数值稳定性检查"""
        batch_size = image.size(0)

        # 输入数据预处理，防止NaN
        image = torch.nan_to_num(image, nan=0.0)
        gene = torch.nan_to_num(gene, nan=0.0)
        tme = torch.nan_to_num(tme, nan=0.0)

        # 特征提取
        image_feat = self.image_mlp(image)  # [batch_size, feature_dim]
        gene_feat = self.gene_snn(gene)  # 使用SNN处理基因特征
        tme_feat = self.tme_mlp(tme)  # [batch_size, feature_dim]

        # 特征裁剪，防止数值爆炸
        image_feat = torch.clamp(image_feat, -10, 10)
        gene_feat = torch.clamp(gene_feat, -10, 10)
        tme_feat = torch.clamp(tme_feat, -10, 10)

        # 应用自注意力机制增强特征
        if image_feat.dim() == 2:
            image_feat = image_feat.unsqueeze(1)
        if gene_feat.dim() == 2:
            gene_feat = gene_feat.unsqueeze(1)
        if tme_feat.dim() == 2:
            tme_feat = tme_feat.unsqueeze(1)

        image_feat = self.image_self_attn(image_feat).squeeze(1)
        gene_feat = self.gene_self_attn(gene_feat).squeeze(1)
        tme_feat = self.tme_self_attn(tme_feat).squeeze(1)

        # 检测gene模态是否缺失（L2范数小于阈值）
        gene_norm = torch.norm(gene_feat, dim=1, keepdim=True)
        gene_present_mask = (gene_norm > self.gene_missing_threshold).float()

        # 首先计算TME和图像的交叉注意力特征
        image2tme_feat = self.cross_attention_gene2tme(
            image_feat.unsqueeze(1),
            tme_feat.unsqueeze(1)
        ).squeeze(1)
        tme2image_feat = self.cross_attention_tme2gene(
            tme_feat.unsqueeze(1),
            image_feat.unsqueeze(1)
        ).squeeze(1)

        # 融合图像和TME特征作为VAE的输入之一
        image_tme_fusion = image2tme_feat + tme2image_feat
        image_tme_fusion = torch.clamp(image_tme_fusion, -10, 10)

        # VAE处理
        # 对融合特征进行编码
        fusion_encoding = self.encoder_tme(image_tme_fusion)
        mu_fusion, logvar_fusion = torch.chunk(fusion_encoding, 2, dim=1)

        # 裁剪mu和logvar
        mu_fusion = torch.clamp(mu_fusion, -10, 10)
        logvar_fusion = torch.clamp(logvar_fusion, -10, 10)

        # 初始化VAE相关变量
        mu_gene = torch.zeros_like(mu_fusion)
        logvar_gene = torch.zeros_like(logvar_fusion)
        mu_combined = torch.zeros_like(mu_fusion)
        logvar_combined = torch.zeros_like(logvar_fusion)

        # 对基因特征进行编码（如果存在）
        if gene_present_mask.sum() > 0:
            # 找出基因存在的样本
            gene_present_indices = gene_present_mask.squeeze() > 0.5
            if gene_present_indices.sum() > 0:
                # 仅对基因存在的样本进行编码
                gene_encoding = self.encoder_gene(gene_feat)
                mu_gene_all, logvar_gene_all = torch.chunk(gene_encoding, 2, dim=1)

                # 裁剪
                mu_gene_all = torch.clamp(mu_gene_all, -10, 10)
                logvar_gene_all = torch.clamp(logvar_gene_all, -10, 10)

                # 将编码结果填充到对应位置
                mu_gene[gene_present_indices] = mu_gene_all[gene_present_indices]
                logvar_gene[gene_present_indices] = logvar_gene_all[gene_present_indices]

                # 对每个样本分别处理
                for i in range(batch_size):
                    if gene_present_mask[i, 0] > 0.5:
                        # 基因存在，使用MoE融合基因特征和融合特征(图像+TME)
                        mu_i, logvar_i = self.mixture_of_experts(
                            [mu_gene[i:i + 1], mu_fusion[i:i + 1]],
                            [logvar_gene[i:i + 1], logvar_fusion[i:i + 1]]
                        )
                        mu_combined[i:i + 1] = mu_i
                        logvar_combined[i:i + 1] = logvar_i
                    else:
                        # 基因缺失，仅使用融合特征编码
                        mu_combined[i:i + 1] = mu_fusion[i:i + 1]
                        logvar_combined[i:i + 1] = logvar_fusion[i:i + 1]
        else:
            # 整个批次都没有基因，仅使用融合特征编码
            mu_combined = mu_fusion
            logvar_combined = logvar_fusion

        # 重参数化采样
        z = self.reparameterize(mu_combined, logvar_combined)
        z = torch.clamp(z, -10, 10)

        # 解码生成新的基因特征
        new_gene_feat = self.decoder(z)
        new_gene_feat = torch.clamp(new_gene_feat, -10, 10)

        # 对基因缺失的样本，使用重建的基因特征
        gene_feat_final = torch.where(gene_present_mask > 0.5, gene_feat, new_gene_feat)

        # 交叉注意力融合（基因和TME）
        gene2tme_feat = self.cross_attention_gene2tme(
            gene_feat_final.unsqueeze(1),
            tme_feat.unsqueeze(1)
        ).squeeze(1)
        tme2gene_feat = self.cross_attention_tme2gene(
            tme_feat.unsqueeze(1),
            gene_feat_final.unsqueeze(1)
        ).squeeze(1)

        # 分子特征融合
        molecular_fusion = gene2tme_feat + tme2gene_feat
        molecular_fusion = torch.clamp(molecular_fusion, -10, 10)

        # 门控注意力融合（图像和分子特征）
        final_fusion, gate_value = self.gated_attention(image_feat, molecular_fusion)
        final_fusion = torch.clamp(final_fusion, -10, 10)

        # 变换图像特征用于MSE损失
        transformed_image_feat = self.image_transform_mlp(image_feat)
        transformed_image_feat = torch.clamp(transformed_image_feat, -10, 10)

        # 生存风险预测
        risk_pred = self.survival_head(final_fusion).squeeze(-1)
        risk_pred = torch.clamp(risk_pred, -10, 10)  # 防止预测值过大

        # 准备VAE训练的目标
        true_fusion_target = gene_feat + molecular_fusion

        return {
            'risk_pred': risk_pred,
            'image_feat': image_feat,
            'gene_feat': gene_feat,
            'gene_feat_final': gene_feat_final,
            'tme_feat': tme_feat,
            'transformed_image_feat': transformed_image_feat,
            'molecular_fusion': molecular_fusion,
            'final_fusion': final_fusion,
            'gate_value': gate_value,
            'mu_gene': mu_gene,
            'logvar_gene': logvar_gene,
            'mu_fusion': mu_fusion,
            'logvar_fusion': logvar_fusion,
            'mu_combined': mu_combined,
            'logvar_combined': logvar_combined,
            'z': z,
            'new_gene_feat': new_gene_feat,
            'gene_present_mask': gene_present_mask,
            'true_fusion_target': true_fusion_target,
            'image_tme_fusion': image_tme_fusion
        }

    def calculate_loss(self, outputs, time, event):
        """计算损失函数，增加数值稳定性检查"""
        # 基本损失函数
        risk_pred = outputs['risk_pred']

        # 检查和处理NaN值
        if torch.isnan(risk_pred).any():
            risk_pred = torch.nan_to_num(risk_pred, nan=0.0)

        # 计算损失前进行裁剪
        risk_pred = torch.clamp(risk_pred, -10, 10)

        try:
            nll_loss = self.nll_loss(risk_pred, time, event)
            rank_loss = self.ranking_loss(risk_pred, time, event)
            mse_loss = self.mse_loss(outputs['final_fusion'], outputs['transformed_image_feat'])
        except Exception as e:
            print(f"Loss calculation error: {e}")
            # 返回安全的损失值
            nll_loss = torch.tensor(0.0, device=time.device)
            rank_loss = torch.tensor(0.0, device=time.device)
            mse_loss = torch.tensor(0.0, device=time.device)

        # 检查损失是否为NaN
        if torch.isnan(nll_loss):
            nll_loss = torch.tensor(0.0, device=time.device)
        if torch.isnan(rank_loss):
            rank_loss = torch.tensor(0.0, device=time.device)
        if torch.isnan(mse_loss):
            mse_loss = torch.tensor(0.0, device=time.device)

        # 基本损失组合
        total_loss, weights = self.dynamic_weight([nll_loss, rank_loss, mse_loss])

        # 初始化VAE相关损失
        recon_loss = torch.tensor(0.0, device=time.device)
        kld_loss = torch.tensor(0.0, device=time.device)
        gene_present_ratio = 0.0

        # 检查当前批次是否有基因模态
        gene_present_mask = outputs.get('gene_present_mask', None)

        # 仅对基因模态存在的样本计算VAE损失
        if gene_present_mask is not None and gene_present_mask.sum() > 0:
            gene_present_ratio = gene_present_mask.mean().item()

            # 找出基因存在的样本索引
            valid_indices = gene_present_mask.squeeze() > 0.5

            if valid_indices.sum() > 0:
                try:
                    # 重建损失
                    recon_loss = F.mse_loss(
                        outputs['new_gene_feat'][valid_indices],
                        outputs['gene_feat'][valid_indices]
                    )

                    # KL散度损失
                    mu_combined = outputs['mu_combined'][valid_indices]
                    logvar_combined = outputs['logvar_combined'][valid_indices]

                    # 裁剪以防止数值不稳定
                    mu_combined = torch.clamp(mu_combined, -10, 10)
                    logvar_combined = torch.clamp(logvar_combined, -10, 10)

                    kld_loss = -0.5 * torch.sum(
                        1 + logvar_combined - mu_combined.pow(2) - logvar_combined.exp()
                    ) / max(1, valid_indices.sum())

                    # 检查并处理NaN
                    if torch.isnan(recon_loss):
                        recon_loss = torch.tensor(0.0, device=time.device)
                    if torch.isnan(kld_loss):
                        kld_loss = torch.tensor(0.0, device=time.device)

                except Exception as e:
                    print(f"VAE loss calculation error: {e}")
                    recon_loss = torch.tensor(0.0, device=time.device)
                    kld_loss = torch.tensor(0.0, device=time.device)

                # VAE损失权重
                vae_recon_weight = 0.1
                vae_kld_weight = 0.01

                # 将VAE损失添加到总损失
                total_loss = total_loss + vae_recon_weight * recon_loss + vae_kld_weight * kld_loss

        # Gate值正则化
        gate_reg_loss = torch.tensor(0.0, device=time.device)
        gate_mean = 0.5

        if 'gate_value' in outputs:
            gate_values = outputs['gate_value']
            gate_mean = torch.mean(gate_values)
            gate_reg_loss = self.gate_reg_alpha * torch.mean((gate_values - 0.5) ** 2)

            if torch.isnan(gate_reg_loss):
                gate_reg_loss = torch.tensor(0.0, device=time.device)
            else:
                total_loss = total_loss + gate_reg_loss

        # 最终检查总损失
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=time.device, requires_grad=True)

        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'rank_loss': rank_loss,
            'mse_loss': mse_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'gate_reg_loss': gate_reg_loss,
            'nll_weight': weights[0],
            'rank_weight': weights[1],
            'mse_weight': weights[2],
            'gate_mean': gate_mean,
            'gene_present_ratio': gene_present_ratio
        }

    def predict(self, image, gene, tme):
        """预测函数"""
        with torch.no_grad():
            outputs = self.forward(image, gene, tme)
            risk_pred = outputs['risk_pred']
            # 确保输出不包含NaN
            risk_pred = torch.nan_to_num(risk_pred, nan=0.0)
            return risk_pred

# def calculate_tauc(risk_scores, survival_times, events, time_points=None, tied_tol=1e-8):
#     """
#     更规范的 tAUC 计算，基于 Heagerty 的动态 AUC + IPCW 方法。
#     使用统一的线性时间点生成策略，避免分位数导致tAUC虚高。
#     修改为使用简单平均而非加权平均。
#     """
#     import numpy as np
#     from lifelines import KaplanMeierFitter
#
#     def calculate_safe_time_points(event_times, censorships):
#         """Calculate safe time points for tAUC evaluation"""
#         uncensored_times = event_times[censorships == 0]
#
#         if len(uncensored_times) == 0:
#             print("Warning: No uncensored events found, using all event times")
#             uncensored_times = event_times
#
#         epsilon = 1.0
#         min_time = np.min(uncensored_times) + epsilon
#         max_time = np.max(uncensored_times) - epsilon
#
#         if min_time >= max_time:
#             print("Warning: Valid time range too small, using single time point")
#             return np.array([(min_time + max_time) / 2])
#
#         return np.linspace(min_time, max_time, num=4)
#
#     risk_scores = np.asarray(risk_scores)
#     survival_times = np.asarray(survival_times)
#     events = np.asarray(events, dtype=np.int64)
#
#     if time_points is None:
#         time_points = calculate_safe_time_points(survival_times, 1 - events)
#
#     try:
#         km_censor = KaplanMeierFitter()
#         km_censor.fit(survival_times, event_observed=1 - events)
#     except Exception as e:
#         print(f"KM拟合失败: {e}")
#         return float('nan')
#
#     try:
#         km_survival = KaplanMeierFitter()
#         km_survival.fit(survival_times, event_observed=events)
#     except Exception as e:
#         print(f"生存函数KM拟合失败: {e}")
#         return float('nan')
#
#     aucs = []
#
#     for t in time_points:
#         is_case = (survival_times <= t) & (events == 1)
#         is_control = (survival_times > t)
#
#         case_idx = np.where(is_case)[0]
#         control_idx = np.where(is_control)[0]
#
#         if len(case_idx) == 0 or len(control_idx) == 0:
#             continue
#
#         G_cases = np.array([
#             km_censor.predict(survival_times[i]) if survival_times[i] <= km_censor.timeline.max()
#             else km_censor.predict(km_censor.timeline.max()) for i in case_idx
#         ])
#         G_controls = np.array([
#             km_censor.predict(min(t, km_censor.timeline.max())) for _ in control_idx
#         ])
#
#         G_cases = np.maximum(G_cases, 1e-6)
#         G_controls = np.maximum(G_controls, 1e-6)
#
#         case_scores = risk_scores[case_idx].reshape(-1, 1)
#         control_scores = risk_scores[control_idx].reshape(1, -1)
#         score_diffs = case_scores - control_scores
#
#         weights_matrix = 1.0 / (G_cases.reshape(-1, 1) * G_controls.reshape(1, -1))
#
#         concordant_mask = score_diffs > tied_tol
#         discordant_mask = score_diffs < -tied_tol
#         tied_mask = np.abs(score_diffs) <= tied_tol
#
#         concordant_weight = np.sum(weights_matrix * concordant_mask)
#         discordant_weight = np.sum(weights_matrix * discordant_mask)
#         tied_weight = np.sum(weights_matrix * tied_mask)
#
#         total_weight = concordant_weight + discordant_weight + tied_weight
#
#         if total_weight > 0:
#             time_auc = (concordant_weight + 0.5 * tied_weight) / total_weight
#             aucs.append(time_auc)
#
#     if len(aucs) == 0:
#         return float('nan')
#
#     # 使用简单平均而非加权平均
#     return np.mean(aucs)


# def calculate_tauc(risk_scores, survival_times, events, time_points=None):
#     """
#     计算 Integrated Brier Score (IBS)
#     IBS越小表示模型预测的校准度越好
#
#     Args:
#         risk_scores: 模型预测的风险分数 (numpy array)
#         survival_times: 实际生存时间 (numpy array)
#         events: 事件指示器，1表示死亡，0表示删失 (numpy array)
#         time_points: 评估的时间点，如果为None则自动生成
#
#     Returns:
#         ibs: Integrated Brier Score (float)
#     """
#     import numpy as np
#     from lifelines import KaplanMeierFitter
#
#     def calculate_safe_time_points(event_times, censorships):
#         """计算安全的时间点用于IBS评估"""
#         uncensored_times = event_times[censorships == 0]
#
#         if len(uncensored_times) == 0:
#             print("Warning: No uncensored events found, using all event times")
#             uncensored_times = event_times
#
#         epsilon = 1.0
#         min_time = np.min(uncensored_times) + epsilon
#         max_time = np.max(uncensored_times) - epsilon
#
#         if min_time >= max_time:
#             print("Warning: Valid time range too small, using single time point")
#             return np.array([(min_time + max_time) / 2])
#
#         # 使用更多时间点以获得更精确的积分估计
#         return np.linspace(min_time, max_time, num=10)
#
#     risk_scores = np.asarray(risk_scores)
#     survival_times = np.asarray(survival_times)
#     events = np.asarray(events, dtype=np.int64)
#
#     # 将风险分数转换为生存概率（假设风险分数越高，生存概率越低）
#     # 使用sigmoid将风险分数归一化到[0,1]
#     predicted_surv_probs = 1.0 / (1.0 + np.exp(risk_scores))
#
#     if time_points is None:
#         time_points = calculate_safe_time_points(survival_times, 1 - events)
#
#     # 拟合删失分布的KM曲线（用于IPCW权重）
#     try:
#         km_censor = KaplanMeierFitter()
#         km_censor.fit(survival_times, event_observed=1 - events)
#     except Exception as e:
#         print(f"KM拟合失败: {e}")
#         return float('nan')
#
#     brier_scores = []
#
#     for t in time_points:
#         # 计算每个样本在时间t的实际生存状态
#         # Y_i(t) = 1 if T_i > t, 0 if T_i <= t
#         actual_survival = (survival_times > t).astype(float)
#
#         # 计算预测的生存概率在时间t的值
#         # 这里简化处理：假设风险分数反映了在整个时间段的累积风险
#         # 更精确的做法是模型直接输出S(t|x)
#         predicted_survival_at_t = predicted_surv_probs
#
#         # 计算IPCW权重
#         weights = np.zeros(len(survival_times))
#
#         for i in range(len(survival_times)):
#             if survival_times[i] <= t and events[i] == 1:
#                 # 事件发生在t之前
#                 G_Ti = km_censor.predict(survival_times[i])
#                 G_Ti = max(G_Ti, 1e-6)  # 避免除零
#                 weights[i] = 1.0 / G_Ti
#             elif survival_times[i] > t:
#                 # 样本在t时刻仍存活
#                 G_t = km_censor.predict(min(t, km_censor.timeline.max()))
#                 G_t = max(G_t, 1e-6)
#                 weights[i] = 1.0 / G_t
#             else:
#                 # 删失时间在t之前，不参与计算
#                 weights[i] = 0.0
#
#         # 计算Brier Score: BS(t) = E[w_i * (Y_i(t) - S(t|X_i))^2]
#         squared_errors = (actual_survival - predicted_survival_at_t) ** 2
#         weighted_squared_errors = weights * squared_errors
#
#         if np.sum(weights) > 0:
#             brier_score_t = np.sum(weighted_squared_errors) / np.sum(weights)
#             brier_scores.append(brier_score_t)
#
#     if len(brier_scores) == 0:
#         return float('nan')
#
#     # IBS是时间积分的Brier Score
#     # 使用梯形法则进行数值积分
#     if len(time_points) > 1:
#         time_range = time_points[-1] - time_points[0]
#         ibs = np.trapz(brier_scores, time_points) / time_range
#     else:
#         ibs = brier_scores[0]
#
#     return ibs


def calculate_tauc(risk_scores, survival_times, events, time_points=None, tied_tol=1e-8):
    """计算时间依赖的AUC (tAUC)，使用IPCW权重和累积/动态AUC方法

    参考：scikit-survival的cumulative_dynamic_auc实现
    """
    import numpy as np
    from lifelines import KaplanMeierFitter
    import pandas as pd

    risk_scores = np.asarray(risk_scores)
    survival_times = np.asarray(survival_times)
    events = np.asarray(events, dtype=np.int64)

    # 自动生成时间点：按照事件时间的百分位数选择4个点
    if time_points is None:
        uncensored_times = survival_times[events == 1]
        if len(uncensored_times) == 0:
            return float('nan')

        # 使用25%, 50%, 75%, 90%百分位数
        percentiles = [25, 50, 75, 90]
        time_points = np.percentile(uncensored_times, percentiles)

        # 确保时间点在有效范围内
        time_points = time_points[time_points < np.max(survival_times)]

        if len(time_points) == 0:
            return float('nan')

    # 估计删失分布的KM曲线（用于IPCW）
    # 注意：这里拟合的是censoring distribution，即把censoring当作event
    km_censor = KaplanMeierFitter()
    km_censor.fit(survival_times, 1 - events)  # 1-events: censoring indicator

    scores = []

    for t in time_points:
        # 定义病例和对照
        # 病例: 在时间t之前或恰好在t时发生事件的样本（且必须是真实事件，不是删失）
        is_case = (survival_times <= t) & (events == 1)
        # 对照: 在时间t之后的样本（存活时间 > t）
        is_control = survival_times > t

        case_indices = np.where(is_case)[0]
        control_indices = np.where(is_control)[0]

        if len(case_indices) == 0 or len(control_indices) == 0:
            continue

        # 计算IPCW权重（仅用于病例）
        case_weights = []
        for i in case_indices:
            G_Ti = km_censor.predict(survival_times[i])
            if isinstance(G_Ti, (pd.Series, pd.DataFrame)):
                G_Ti = float(G_Ti.iloc[0])
            else:
                G_Ti = float(G_Ti)

            if G_Ti <= 0:
                case_weights.append(0.0)
            else:
                case_weights.append(1.0 / G_Ti)

        case_weights = np.array(case_weights)

        # 计算concordance
        numerator = 0.0

        for idx, i in enumerate(case_indices):
            weight_i = case_weights[idx]
            if weight_i == 0:
                continue

            for j in control_indices:
                # 判断concordance
                if abs(risk_scores[i] - risk_scores[j]) <= tied_tol:
                    # 相同风险分数，算0.5
                    numerator += 0.5 * weight_i
                elif risk_scores[i] > risk_scores[j]:
                    # 高风险分数应对应更短的生存时间
                    numerator += weight_i

        # 分母 = (对照数) × (加权病例数)
        denominator = len(control_indices) * np.sum(case_weights)

        if denominator == 0:
            continue

        auc_t = numerator / denominator
        scores.append((t, auc_t))

    if len(scores) == 0:
        return float('nan')

    # 提取时间点和对应的AUC值
    times, aucs = zip(*scores)
    times = np.array(times)
    aucs = np.array(aucs)

    # 使用事件分布的KM曲线计算权重
    km_events = KaplanMeierFitter()
    km_events.fit(survival_times, events)

    # 获取生存概率
    survival_probs = km_events.predict(times)
    if isinstance(survival_probs, (pd.Series, pd.DataFrame)):
        survival_probs = survival_probs.values.flatten()
    else:
        survival_probs = np.asarray(survival_probs)

    # 计算权重：基于生存函数的变化 w_i = S(t_{i-1}) - S(t_i)
    weights = -np.diff(np.concatenate([[1.0], survival_probs]))

    # 归一化：除以总的概率变化量
    weight_sum = 1.0 - survival_probs[-1]

    if weight_sum <= 0 or weight_sum > 1.0:
        # 如果无法正确归一化，使用简单平均
        mean_auc = np.mean(aucs)
    else:
        mean_auc = np.sum(aucs * weights) / weight_sum

    return mean_auc


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    all_risk_preds = []
    all_times = []
    all_events = []
    all_loss_info = {
        'total_loss': 0, 'nll_loss': 0, 'rank_loss': 0,
        'mse_loss': 0, 'recon_loss': 0, 'kld_loss': 0
    }
    all_weight_info = {
        'nll_weight': 0, 'rank_weight': 0, 'mse_weight': 0,
        'recon_weight': 0, 'kld_weight': 0
    }
    gene_present_ratio_sum = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            image = batch['image'].to(device)
            gene = batch['gene'].to(device)
            tme = batch['TME'].to(device)
            time = batch['time'].to(device)
            event = batch['event'].to(device)
            # gene_new = torch.zeros_like(gene)  # 将这些样本的基因数据置为零
            # 前向传播
            outputs = model(image, gene, tme)
            loss_info = model.calculate_loss(outputs, time, event)

            # 收集预测和标签a
            all_risk_preds.append(outputs['risk_pred'].cpu().numpy())
            all_times.append(time.cpu().numpy())
            all_events.append(event.cpu().numpy())

            # 累积损失和权重
            for key in all_loss_info:
                if key in loss_info:
                    all_loss_info[key] += loss_info[key].item() if torch.is_tensor(loss_info[key]) else loss_info[key]

            # 累积权重
            for key in all_weight_info:
                if key in loss_info:
                    all_weight_info[key] += loss_info[key]

            # 记录基因存在比例
            if 'gene_present_ratio' in loss_info:
                gene_present_ratio_sum += loss_info['gene_present_ratio']

            count += 1

    # 计算平均损失和权重
    for key in all_loss_info:
        all_loss_info[key] /= count

    for key in all_weight_info:
        all_weight_info[key] /= count

    gene_present_ratio = gene_present_ratio_sum / count

    all_risk_preds = np.concatenate(all_risk_preds)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    c_index = concordance_index(all_times, -all_risk_preds, all_events)

    # 计算时间依赖的AUC (tAUC)
    tauc = calculate_tauc(all_risk_preds, all_times, all_events)

    # 添加评估指标到结果
    all_loss_info.update({
        'c_index': c_index,
        'tauc': tauc,
        'gene_present_ratio': gene_present_ratio
    })

    # 添加权重到结果
    for key, value in all_weight_info.items():
        all_loss_info[key] = value

    return all_loss_info
