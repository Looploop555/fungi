import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def feature_augmentation(feature, aug_type='gaussian_dropout', noise_std=0.05, dropout_prob=0.1):
    if aug_type == 'gaussian_dropout':
        noise = torch.randn_like(feature) * noise_std
        feature = feature + noise
        mask = (torch.rand_like(feature) < dropout_prob).float()
        feature = feature * (1 - mask)
    elif aug_type == 'gaussian':
        noise = torch.randn_like(feature) * noise_std
        feature = feature + noise
    elif aug_type == 'dropout':
        mask = (torch.rand_like(feature) < dropout_prob).float()
        feature = feature * (1 - mask)
    return feature


def model_save(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model


def get_topk(data, topk=10):
    static = np.zeros(topk, dtype=np.int64)
    pred = np.argsort(data, axis=1, reversed=True)
    for i, p in enumerate(pred):
        tk = p.tolist().index(i)
        static[min(tk, topk):] += 1
    return 0


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def train_process(pkl_dir, min_sample=4, val=False):
    train_data = []
    for key in tqdm([path for path in os.listdir(pkl_dir)]):
        value = load_pkl(os.path.join(pkl_dir, key))
        if val or len(value) > min_sample:
            train_data.append(os.path.join(pkl_dir, key))
        # train_data.append(os.path.join(pkl_dir, key))
    return train_data


def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def sample_feature(pkl, sample_nums=128):
    sample_data = []
    sample_index = np.random.choice(len(pkl), sample_nums, replace=False)
    for key in sample_index:
        pkl_value = load_pkl(pkl[key])
        index = np.random.choice(len(pkl_value), 2, replace=False)
        sample_data.append([pkl_value[index[0]], pkl_value[index[1]]])
    sample_data = np.array(sample_data)
    return sample_data


class MyDataset(Dataset):
    def __init__(self, data_dir, max_sample=4, device='cuda'):
        self.max_sample = max_sample
        data = train_process(data_dir, max_sample, False)
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pkl = self.data[idx]
        sample_data = load_pkl(pkl)
        index = np.random.choice(len(sample_data), min(self.max_sample, len(sample_data)), replace=False)
        return torch.stack([torch.tensor(sample_data[i]) for i in index], dim=0).to(self.device).to(torch.float32)


class PatchMocoAligner(nn.Module):
    """
    Processes a sequence of patch features using a Transformer Encoder
    and a class token for feature alignment/fine-tuning.

    Input shape: (B, P, C_in) where B=batch size, P=num_patches, C_in=input_channel_dim
    Output shape: (B, C_out) where C_out=output_channel_dim
    """

    def __init__(self,
                 in_dim,  # Input feature dim (C_in) per patch
                 out_dim,  # Final output feature dim (C_out)
                 patch_count=196,  # Number of patches (P)
                 embed_dim=768,  # Internal embedding dimension for Transformer
                 depth=6,  # Number of Transformer Encoder layers
                 num_heads=12,  # Number of attention heads
                 mlp_ratio=4.0,  # Ratio for MLP hidden dim in Transformer Encoder
                 dropout=0.0,
                 pool='cls',  # Pooling method for final output: 'cls' or 'mean'
                 moco_dim=None):  # Optional: Intermediate dim like in original SampleMoco
        super().__init__()
        self.patch_count = patch_count
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.pool = pool

        # 1. Linear projection for input patches (optional but common)
        # Maps C_in -> embed_dim
        self.patch_embed = nn.Linear(in_dim, embed_dim)

        # 2. Class token
        # Learnable parameter prepended to the sequence of patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)  # Standard ViT initialization

        # 3. Positional embedding
        # Learnable embeddings for class token + all patch tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + patch_count, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)  # Standard ViT initialization
        self.pos_drop = nn.Dropout(p=dropout)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",  # Common activation in Transformers
            batch_first=True  # Important: expects (B, Seq, Feat) input
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth
        )

        # 5. Layer Normalization (applied after Transformer, before head)
        self.norm = nn.LayerNorm(embed_dim)

        # 6. Output Head (maps the class token embedding to the final out_dim)
        # You can choose a simple linear layer or replicate the original SampleMoco structure
        if moco_dim is not None and moco_dim > 0:
            # Replicates the SampleMoco structure more closely
            self.mlp_head = nn.Sequential(
                nn.Linear(embed_dim, moco_dim),
                # Consider adding an activation/norm here if needed
                nn.GELU(),
                nn.LayerNorm(moco_dim),
                nn.Linear(moco_dim, out_dim)
            )
        else:
            # Simpler direct mapping
            self.mlp_head = nn.Linear(embed_dim, out_dim)

        # Initialize weights for linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, P, C_in)

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out)
        """
        x = x.squeeze(1)  # (B, P, C_in)
        B, P, C_in = x.shape
        assert P == self.patch_count, \
            f"Input patch count ({P}) doesn't match model configuration ({self.patch_count})"

        # 1. Embed patches
        x = self.patch_embed(x)  # (B, P, embed_dim)

        # 2. Prepend class token
        # Expand cls_token to batch size: (1, 1, embed_dim) -> (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + P, embed_dim)

        # 3. Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # (B, 1 + P, embed_dim)

        if self.pool == 'cls':
            # 5. Extract class token representation (the first token)
            cls_output = x[:, 0]  # (B, embed_dim)

            # 6. Apply Layer Normalization
            cls_output = self.norm(cls_output)

            # 7. Pass through MLP head
            output = self.mlp_head(cls_output)  # (B, C_out)
            # output = output.unsqueeze(1)  # (B, 1, C_out)

        else:
            mean_output = x.mean(dim=1)
            output = self.norm(mean_output)
            output = self.mlp_head(output)  # (B, C_out)
        return output


class PairwiseContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(PairwiseContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        # print(f"Initialized PairwiseContrastiveLoss with temperature={temperature}")

    def forward(self, pred, gt):
        device = pred.device
        batch_size = pred.shape[0]

        pred_normalized = F.normalize(pred, p=2, dim=1)

        similarity_matrix = torch.matmul(pred_normalized, pred_normalized.T)

        mask_diag = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        mask_positives = (gt == 1).bool() & mask_diag
        # shape: (B, B)

        # mask_negatives = (gt == 0).bool() & mask_diag # SupCon

。

logits = similarity_matrix / self.temperature
# shape: (B, B)

logits_masked_diag = logits * mask_diag.float()
logits_max, _ = torch.max(logits_masked_diag, dim=1, keepdim=True)
logits_stable = logits - logits_max.detach()
# shape: (B, B)

exp_logits_stable = torch.exp(logits_stable) * mask_diag.float(
    log_prob_denominator=torch.log(exp_logits_stable.sum(dim=1, keepdim=True) + 1e-8)
# shape: (B, 1)

#  log [ exp(sim(i, p)/T) / Σ_{k ∈ A(i)} exp(sim(i, k)/T) ]
# = (sim(i, p)/T - max_logit_i) - log(Σ_{k ∈ A(i)} exp(sim(i, k)/T - max_logit_i))
# = logits_stable[i, p] - log_prob_denominator[i]
log_prob = logits_stable - log_prob_denominator
# shape: (B, B)

# L = Σ_i (1 / |P(i)|) * Σ_{p ∈ P(i)} -log_prob(i, p)


num_positives_per_anchor = mask_positives.float().sum(dim=1)
# shape: (B,)

valid_anchors_mask = num_positives_per_anchor > 0
if not valid_anchors_mask.any():
    print("Warning: No positive pairs found in the batch (excluding self). Returning 0 loss.")
return torch.tensor(0.0, device=device, requires_grad=True)

masked_log_prob = -log_prob * mask_positives.float()
# shape: (B, B)

sum_neg_log_prob_per_anchor = masked_log_prob.sum(dim=1)
# shape: (B,)


loss_per_anchor = sum_neg_log_prob_per_anchor / torch.clamp(num_positives_per_anchor, min=1e-8)

# loss = loss_per_anchor[valid_anchors_mask].mean()
total_loss = loss_per_anchor.sum()
num_valid_anchors = valid_anchors_mask.float().sum()

loss = total_loss / torch.clamp(num_valid_anchors, min=1e-8)

return loss


def train(model, data, optimizer, scheduler, loss_func):
    model.train()
    optimizer.zero_grad()
    B, T, P, C = data.shape
    data = data.view(B * T, P, C)
    gt = torch.zeros(B, T, B, T).to(data.device)
    for i in range(B):
        gt[i, :, i, :] = 1
    gt = gt.view(B * T, B * T)
    output = model(data)
    loss = loss_func(output, gt)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def inference(model, test_data, device):
    model.eval()
    infer_info = {}
    with torch.no_grad():
        for file_path in tqdm(test_data):
            data = torch.tensor(np.array(load_pkl(file_path)), dtype=torch.float32).to(device)
            output = model(data)
            output /= output.norm(dim=-1, keepdim=True)
            infer_info[os.path.basename(file_path[:-4])] = output.detach().cpu().numpy()
    return infer_info


def model_train(model, lr, batch_size, max_lens, epochs, data_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = MyDataset(data_path, max_lens)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True,
                                          multiprocessing_context='spawn')
    step = epochs * len(dataset)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=step, T_mult=1, eta_min=0,
                                                                     last_epoch=-1)
    with tqdm(range(epochs), desc=('total step:%08d' % step)) as pbar:
        for epochs in pbar:
            train_loss = 0
            for data in dataset:
                train_loss += train(model, data, optimizer, scheduler, PairwiseContrastiveLoss())
            pbar.set_postfix(train_loss=train_loss / len(dataset), lr=scheduler.get_last_lr()[0])
            model_save(model, model_save_path)
    return model


if __name__ == "__main__":
    device = 'cuda'
    pool = 'cls'

    # cls Large_width
    model = PatchMocoAligner(
        in_dim=1536,
        out_dim=1536,
        patch_count=261,
        embed_dim=2048,
        depth=1,
        num_heads=16,
        mlp_ratio=2,
        pool=pool,
    )
    model_save_path = 'model.pkl'
    if os.path.exists(model_save_path):
        model = load_model(model, model_save_path)
    model.to(device)

    model = model_train(
        model,
        lr=0.0002,
        batch_size=1024,
        max_lens=2,
        epochs=500,
        data_path='/data/train_features'
    )

    infer_train_info = inference(model, train_process('/data/train_features', val=True), device)
    save_pkl(infer_train_info, '/data/new/train_feature.pkl')

    infer_val_info = inference(model, train_process('/data/val_features', val=True), device)
    save_pkl(infer_val_info, '/data/new/val_feature.pkl')

    infer_test_info = inference(model, train_process('/data/test_features', val=True), device)
    save_pkl(infer_test_info, '/data/new/test_feature.pkl')
