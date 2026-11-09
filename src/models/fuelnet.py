"""
FuelNet — Deep Residual MLP (Phase 4)
Exact architecture:
- make embedding vocab x 32, model embedding vocab x 64
- Fusion: concat [make_emb, model_emb, numeric_features (44 = 5 scaled + 39 ohe)] -> Linear 256 -> LayerNorm -> GELU
- 4 Residual Blocks: x = x + Dropout( Linear2( GELU( Linear1( LayerNorm(x) ) ) ) ), dropout 0.1
- Output head: Linear 256 -> 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MAPPINGS = PROJECT_ROOT / "models_artifacts" / "category_mappings.json"

class ResidualBlock(nn.Module):
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = x + drop(Linear2(GELU(Linear1(LayerNorm(x)))))
        residual = self.ln(x)
        residual = self.lin1(residual)
        residual = F.gelu(residual)
        residual = self.lin2(residual)
        residual = self.dropout(residual)
        return x + residual

class FuelNet(nn.Module):
    def __init__(self,
                 make_vocab_size,
                 model_vocab_size,
                 numeric_dim=44,  # 5 scaled numeric + 39 one-hot (from Phase 2)
                 make_emb_dim=32,
                 model_emb_dim=64,
                 hidden_dim=256,
                 n_blocks=4,
                 dropout=0.1):
        super().__init__()
        self.make_vocab_size = make_vocab_size
        self.model_vocab_size = model_vocab_size
        self.numeric_dim = numeric_dim
        self.make_emb_dim = make_emb_dim
        self.model_emb_dim = model_emb_dim
        self.hidden_dim = hidden_dim

        self.make_emb = nn.Embedding(make_vocab_size, make_emb_dim)
        self.model_emb = nn.Embedding(model_vocab_size, model_emb_dim)

        fusion_in = make_emb_dim + model_emb_dim + numeric_dim
        self.fusion_linear = nn.Linear(fusion_in, hidden_dim)
        self.fusion_ln = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])

        self.head = nn.Linear(hidden_dim, 4)  # fuel_city, fuel_hwy, fuel_comb, co2

        self._init_weights()

    def _init_weights(self):
        # Xavier/Glorot init for linears, normal for embeddings
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    def forward(self, numeric_features, make_idx, model_idx):
        """
        numeric_features: FloatTensor (B, numeric_dim) — already scaled + one-hot concatenated (44)
        make_idx: LongTensor (B,) — embedding index for make (includes unknown)
        model_idx: LongTensor (B,) — embedding index for model
        Returns: (B, 4) predictions
        """
        make_e = self.make_emb(make_idx)  # (B, 32)
        model_e = self.model_emb(model_idx)  # (B, 64)
        # Fusion
        x = torch.cat([make_e, model_e, numeric_features], dim=1)  # (B, 140)
        x = self.fusion_linear(x)
        x = self.fusion_ln(x)
        x = F.gelu(x)
        for blk in self.blocks:
            x = blk(x)
        out = self.head(x)  # (B, 4)
        return out

    @classmethod
    def from_mappings(cls, mappings_path=DEFAULT_MAPPINGS, numeric_dim=44, **kwargs):
        with open(mappings_path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        return cls(
            make_vocab_size=m['make_vocab_size'],
            model_vocab_size=m['model_vocab_size'],
            numeric_dim=numeric_dim,
            **kwargs
        )

if __name__ == "__main__":
    # Quick sanity check
    import json
    mappings_path = PROJECT_ROOT / "models_artifacts" / "category_mappings.json"
    with open(mappings_path) as f:
        m = json.load(f)
    print(f"Make vocab {m['make_vocab_size']}, model vocab {m['model_vocab_size']}")
    model = FuelNet(make_vocab_size=m['make_vocab_size'], model_vocab_size=m['model_vocab_size'], numeric_dim=44)
    print(model)
    # Test forward
    B = 4
    num = torch.randn(B, 44)
    make_idx = torch.randint(0, m['make_vocab_size'], (B,))
    model_idx = torch.randint(0, m['model_vocab_size'], (B,))
    out = model(num, make_idx, model_idx)
    print(f"Forward output shape: {out.shape} (expected [4,4])")
    print(out)
    # Param count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
