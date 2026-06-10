from __future__ import annotations

import torch
from torch import nn
from transformers import BertConfig, BertModel


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 2,
        kernel_sizes=(3, 4, 5),
        num_filters: int = 128,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, num_filters, kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids).transpose(1, 2)
        features = []
        for conv in self.convs:
            activated = torch.relu(conv(x))
            features.append(torch.max(activated, dim=2).values)
        logits = self.classifier(self.dropout(torch.cat(features, dim=1)))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class SelfAttentionPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states, content_mask):
        scores = self.scorer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~content_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class SLKRoBERTa(nn.Module):
    """Chinese RoBERTa-WWM with optional multi-granularity pooling and lexicon fusion."""

    model_type = "slk_roberta"

    def __init__(
        self,
        pretrained_dir=None,
        config: BertConfig | None = None,
        lexicon_dim: int = 6,
        fusion_dim: int = 256,
        dropout: float = 0.2,
        use_multi_pooling: bool = True,
        use_gated_fusion: bool = True,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        super().__init__()
        if pretrained_dir is not None:
            self.bert = BertModel.from_pretrained(str(pretrained_dir))
            config = self.bert.config
        elif config is not None:
            self.bert = BertModel(config)
        else:
            raise ValueError("pretrained_dir or config is required")

        self.config = config
        self.lexicon_dim = lexicon_dim
        self.fusion_dim = fusion_dim
        self.dropout_rate = dropout
        self.use_multi_pooling = use_multi_pooling
        self.use_gated_fusion = use_gated_fusion
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

        hidden_size = config.hidden_size
        self.attention_pool = SelfAttentionPool(hidden_size)
        semantic_input_size = hidden_size * 3 if use_multi_pooling else hidden_size
        self.semantic_projection = nn.Sequential(
            nn.Linear(semantic_input_size, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )
        self.lexicon_projection = nn.Sequential(
            nn.Linear(lexicon_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.shared_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim, 2)

    def _content_mask(self, input_ids, attention_mask):
        mask = attention_mask.bool()
        for special_id in {self.cls_token_id, self.sep_token_id, self.pad_token_id}:
            if special_id is not None:
                mask = mask & input_ids.ne(special_id)
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = True
        return mask

    def _multi_granularity_pool(self, hidden_states, content_mask):
        mask = content_mask.unsqueeze(-1).to(hidden_states.dtype)
        mean_pool = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        masked_hidden = hidden_states.masked_fill(
            ~content_mask.unsqueeze(-1), torch.finfo(hidden_states.dtype).min
        )
        max_pool = masked_hidden.max(dim=1).values
        attention_pool = self.attention_pool(hidden_states, content_mask)
        return torch.cat([mean_pool, max_pool, attention_pool], dim=-1)

    def forward(
        self,
        input_ids,
        attention_mask,
        lexicon_features=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if self.use_multi_pooling:
            content_mask = self._content_mask(input_ids, attention_mask)
            pooled = self._multi_granularity_pool(outputs.last_hidden_state, content_mask)
        else:
            pooled = outputs.last_hidden_state[:, 0]
        semantic = self.semantic_projection(pooled)

        gate = None
        if self.use_gated_fusion:
            if lexicon_features is None:
                raise ValueError("lexicon_features are required when gated fusion is enabled")
            lexicon = self.lexicon_projection(lexicon_features.to(semantic.dtype))
            gate = self.fusion_gate(torch.cat([semantic, lexicon], dim=-1))
            fused = gate * semantic + (1.0 - gate) * lexicon
        else:
            fused = semantic
        shared = self.shared_layer(fused)

        logits = self.classifier(shared)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None

        return {
            "loss": loss,
            "logits": logits,
            "gate": gate,
        }


class FGM:
    """Fast Gradient Method restricted to word embedding weights."""

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1.0,
        parameter_suffix: str = "word_embeddings.weight",
    ):
        self.model = model
        self.epsilon = epsilon
        self.parameter_suffix = parameter_suffix
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and name.endswith(self.parameter_suffix)
                and param.grad is not None
            ):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
