import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec - max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = masked_sums == 0
    masked_sums += zeros.float()
    return masked_exps / masked_sums


def masked_argmax(vec, mask, dim, keepdim=False):
    vec_rank = torch.argsort(torch.argsort(vec, dim=dim), dim=dim) + 1
    masked_vec_rank = vec_rank * mask.float()
    return torch.argmax(masked_vec_rank, dim=dim, keepdim=keepdim)


def masked_sum(vec, mask, dim, keepdim=False):
    return torch.sum(vec * mask.float(), dim=dim, keepdim=keepdim)


def masked_mean(vec, mask, dim, keepdim=False):
    return masked_sum(vec, mask, dim, keepdim) / torch.sum(
        mask.float(), dim=dim, keepdim=keepdim
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head, k_dim, v_dim):
        super().__init__()
        self.n_head = n_head
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.query = nn.Linear(embed_dim, n_head * k_dim)
        self.key = nn.Linear(embed_dim, n_head * k_dim)
        self.value = nn.Linear(embed_dim, n_head * v_dim)
        self.out = nn.Linear(n_head * v_dim, embed_dim)

    def forward(self, h, attention_mask):
        batch, seq_len, _ = h.shape
        q = self.query(h).view(batch, seq_len, self.n_head, self.k_dim)
        k = self.key(h).view(batch, seq_len, self.n_head, self.k_dim)
        v = self.value(h).view(batch, seq_len, self.n_head, self.v_dim)
        q = q.permute(0, 2, 1, 3)  # [b, h, l, d]
        k = k.permute(0, 2, 3, 1)  # [b, h, d, l]
        a = torch.matmul(q, k) / math.sqrt(self.k_dim)  # [b, h, l, l]
        a = masked_softmax(a, attention_mask[:, None, None, :], dim=3)  # [b, h, l, l]
        o = torch.matmul(a, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous()
        # [b, h, l, l] x [b, h, l, d] = [b, h, l, d] -> [b, l, h, d]
        o = o.view(batch, seq_len, -1)  # [b, l, h*d]
        o = self.out(o)  # [b, l, h*d]
        return o, a


class Bert(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        padding_idx=0,
        max_length=512,
        drop_prob=0.1,
        n_head=12,
        k_dim=64,
        v_dim=64,
        feedforward_dim=3072,
        n_layer=12,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.position_embeddings = nn.Embedding(
            max_length, embed_dim, padding_idx=padding_idx
        )
        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        self.embedding_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mhsa": MultiHeadSelfAttention(embed_dim, n_head, k_dim, v_dim),
                        "norm": nn.LayerNorm(embed_dim, eps=1e-12),
                        "ff": nn.Sequential(
                            nn.Linear(embed_dim, feedforward_dim),
                            nn.GELU(),
                            nn.Linear(feedforward_dim, embed_dim),
                        ),
                        "ff_norm": nn.LayerNorm(embed_dim, eps=1e-12),
                    }
                )
                for _ in range(n_layer)
            ]
        )
        self.dropout = nn.Dropout(p=drop_prob)
        self.padding_idx = padding_idx
        self.n_head = n_head
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            h = self.forward_embedding(input_ids)
        for i, module_dict in enumerate(self.encoder):
            h = self.forward_layer(h, attention_mask, module_dict)
        return h

    def forward_embedding(self, input_ids):
        batch, seq_len = input_ids.shape
        word = self.word_embeddings(input_ids)
        position_ids = torch.arange(0, seq_len, device=input_ids.device)
        position_ids = position_ids[None, :].expand(batch, -1)
        position = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(position_ids)
        token_type = self.token_type_embeddings(token_type_ids)
        # h = self.dropout(self.embedding_norm(word + position + token_type))
        h = self.embedding_norm(word + position + token_type)
        return h

    def forward_layer(self, h, attention_mask, module_dict):
        h = module_dict["norm"](
            h + self.dropout(module_dict["mhsa"](h, attention_mask)[0])
        )
        h = module_dict["ff_norm"](h + self.dropout(module_dict["ff"](h)))
        return h

    def load(self):
        model = BertModel.from_pretrained("bert-base-uncased")
        self.word_embeddings.load_state_dict(
            model.embeddings.word_embeddings.state_dict()
        )
        self.position_embeddings.load_state_dict(
            model.embeddings.position_embeddings.state_dict()
        )
        self.token_type_embeddings.load_state_dict(
            model.embeddings.token_type_embeddings.state_dict()
        )
        self.embedding_norm.load_state_dict(model.embeddings.LayerNorm.state_dict())
        for t, f in zip(self.encoder, model.encoder.layer):
            t["mhsa"].query.load_state_dict(f.attention.self.query.state_dict())
            t["mhsa"].key.load_state_dict(f.attention.self.key.state_dict())
            t["mhsa"].value.load_state_dict(f.attention.self.value.state_dict())
            t["mhsa"].out.load_state_dict(f.attention.output.dense.state_dict())
            t["norm"].load_state_dict(f.attention.output.LayerNorm.state_dict())
            t["ff"][0].load_state_dict(f.intermediate.dense.state_dict())
            t["ff"][2].load_state_dict(f.output.dense.state_dict())
            t["ff_norm"].load_state_dict(f.output.LayerNorm.state_dict())


class TMix(nn.Module):
    def __init__(self, embedding_model, mixup_layer=0):
        super().__init__()
        self.embedding_model = embedding_model
        self.mixup_layer = mixup_layer

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        with torch.no_grad():
            h = self.embedding_model.forward_embedding(input_ids)
        for module_dict in self.embedding_model.encoder[: self.mixup_layer]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict)
        if mixup_indices is not None:
            h = lambda_ * h + (1 - lambda_) * h[mixup_indices]
        for module_dict in self.embedding_model.encoder[self.mixup_layer :]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict)
        return h


class NonlinearMix(nn.Module):
    def __init__(self, embedding_model, mixup_layer=0, max_length=256, d_label=100):
        super().__init__()
        self.embedding_model = embedding_model
        self.mixup_layer = mixup_layer
        self.policy_mapping_f = nn.Sequential(
            nn.Linear(embedding_model.embed_dim * max_length, d_label), nn.Sigmoid()
        )
        self.max_length = max_length

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        with torch.no_grad():
            h = self.embedding_model.forward_embedding(input_ids)
        for module_dict in self.embedding_model.encoder[: self.mixup_layer]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict)
        if mixup_indices is not None:
            assert all(x == y for x, y in zip(h.shape, lambda_.shape))
            h = lambda_ * h + (1 - lambda_) * h[mixup_indices]
            policy_input = F.pad(h, pad=(0, 0, 0, self.max_length - h.shape[1]))
            policy_input = policy_input.view(policy_input.shape[0], -1)
            # Policy mapping function F for mixing label embedding vector
            phi = self.policy_mapping_f(policy_input)
        for module_dict in self.embedding_model.encoder[self.mixup_layer :]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict)
        if mixup_indices is None:
            return h
        else:
            return h, phi


class EmbeddingGenerator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, 12, 64, 64)
        self.layer = nn.Sequential(
            nn.Linear(2 * embed_dim, 128), nn.Tanh(), nn.Linear(128, 3), nn.Softmax()
        )
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h, attention_mask, mixup_indices, eps):
        h = self.norm(h + self.dropout(self.mhsa(h, attention_mask)[0]))
        sentence_h = masked_mean(h, attention_mask[:, :, None], dim=1)
        mix_sentence_h = torch.cat((sentence_h, sentence_h[mixup_indices]), dim=1)
        outputs = self.layer(mix_sentence_h)
        # logging.info("alpha: %.4f, Delta: %.4f" % (outputs[:, 0].mean(), outputs[:, 1].mean()))
        return outputs[:, 1] * eps + outputs[:, 0]


class ManifoldDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, 12, 64, 64)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h, attention_mask):
        h = self.norm(h + self.dropout(self.mhsa(h, attention_mask)[0]))
        h = masked_mean(h, attention_mask[:, :, None], dim=1)
        return self.classifier(h)


class OoMMix(nn.Module):
    def __init__(self, embedding_model, g_layer=0, d_layer=0):
        super().__init__()
        self.embedding_model = embedding_model
        self.embedding_generator = EmbeddingGenerator(embedding_model.embed_dim)
        self.manifold_discriminator = ManifoldDiscriminator(embedding_model.embed_dim)
        self.g_layer = g_layer
        self.d_layer = d_layer
        assert g_layer <= d_layer

    def forward(self, input_ids, attention_mask, mixup_indices=None, eps=None):
        with torch.no_grad():
            h = self.embedding_model.forward_embedding(input_ids)
        for layer_idx, module_dict in enumerate(self.embedding_model.encoder):
            if mixup_indices is not None:
                if layer_idx == self.g_layer:
                    # Generate mixing coefficient
                    gamma = self.embedding_generator(
                        h.detach(), attention_mask.detach(), mixup_indices, eps
                    )  # [B]
                    mix_h = (
                        gamma[:, None, None] * h.detach()
                        + (1 - gamma)[:, None, None] * h[mixup_indices].detach()
                    )
                    mix_h = torch.where(
                        attention_mask[:, :, None]
                        & attention_mask[mixup_indices, :, None],
                        mix_h,
                        h.detach(),
                    )
                if layer_idx == self.d_layer:
                    # Manifold Discriminator
                    pos = self.manifold_discriminator(h, attention_mask)
                    neg = self.manifold_discriminator(mix_h, attention_mask)
                    output = torch.cat((pos, neg), dim=0)
                    label = torch.cat(
                        (torch.ones_like(pos), torch.zeros_like(neg)), dim=0
                    )
                    intr_loss = F.binary_cross_entropy_with_logits(output, label)
                if layer_idx >= self.g_layer:
                    mix_h = self.embedding_model.forward_layer(
                        mix_h, attention_mask, module_dict
                    )
            if layer_idx >= self.g_layer:
                h = self.embedding_model.forward_layer(h, attention_mask, module_dict)
            else:
                with torch.no_grad():
                    h = self.embedding_model.forward_layer(
                        h, attention_mask, module_dict
                    )
                # h = self.embedding_model.forward_layer(h, attention_mask, module_dict)
        if mixup_indices is not None and 12 == self.d_layer:
            # Manifold Discriminator
            pos = self.manifold_discriminator(h, attention_mask)
            neg = self.manifold_discriminator(mix_h, attention_mask)
            output = torch.cat((pos, neg), dim=0)
            label = torch.cat((torch.ones_like(pos), torch.zeros_like(neg)), dim=0)
            intr_loss = F.binary_cross_entropy_with_logits(output, label)
        if mixup_indices is None:
            return h
        else:
            return h, mix_h, gamma, intr_loss

    def predict(self, input_ids, attention_mask):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask)


def create_sentence_classifier(embed_dim, n_class):
    return nn.Sequential(nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, n_class))


class SentenceClassificationModel(nn.Module):
    def __init__(self, embedding_model, n_class):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = create_sentence_classifier(embedding_model.embed_dim, n_class)

    def forward(self, input_ids, attention_mask):
        h = self.embedding_model(input_ids, attention_mask)
        return self.classifier(masked_mean(h, attention_mask[:, :, None], dim=1))

    def predict(self, input_ids, attention_mask):
        out = self.forward(input_ids, attention_mask)
        return out.argmax(dim=1)

    def load(self):
        self.embedding_model.load()

    def get_embedding_model(self):
        return self.embedding_model


class TMixSentenceClassificationModel(nn.Module):
    def __init__(self, embedding_model, mixup_layer, n_class):
        super().__init__()
        self.mix_model = TMix(embedding_model, mixup_layer=mixup_layer)
        self.classifier = create_sentence_classifier(embedding_model.embed_dim, n_class)
        self.sentence_h = nn.Identity()

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        h = self.mix_model(
            input_ids, attention_mask, mixup_indices=mixup_indices, lambda_=lambda_
        )
        h = self.sentence_h(masked_mean(h, attention_mask[:, :, None], dim=1))
        return self.classifier(h)

    def predict(self, input_ids, attention_mask):
        out = self.forward(input_ids, attention_mask)
        return out.argmax(dim=1)

    def load(self):
        self.mix_model.embedding_model.load()

    def get_embedding_model(self):
        return self.mix_model.embedding_model


class MixupTransformerSentenceClassificationModel(nn.Module):
    def __init__(self, embedding_model, n_class):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = create_sentence_classifier(embedding_model.embed_dim, n_class)

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        h = self.embedding_model(input_ids, attention_mask)
        h = masked_mean(h, attention_mask[:, :, None], dim=1)
        if mixup_indices is not None:
            h = lambda_ * h + (1 - lambda_) * h[mixup_indices]
        return self.classifier(h)

    def predict(self, input_ids, attention_mask):
        out = self.forward(input_ids, attention_mask)
        return out.argmax(dim=1)

    def load(self):
        self.embedding_model.load()

    def get_embedding_model(self):
        return self.embedding_model


class NonlinearMixSentenceClassificationModel(nn.Module):
    def __init__(self, embedding_model, mixup_layer, max_length, d_class, n_class):
        super().__init__()
        self.mix_model = NonlinearMix(
            embedding_model,
            mixup_layer=mixup_layer,
            max_length=max_length,
            d_label=d_class,
        )
        self.classifier = create_sentence_classifier(embedding_model.embed_dim, d_class)
        self.label_matrix = nn.Parameter(torch.randn(n_class, d_class))

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        if mixup_indices is None:
            h = self.mix_model(
                input_ids, attention_mask, mixup_indices=mixup_indices, lambda_=lambda_
            )
            return self.classifier(masked_mean(h, attention_mask[:, :, None], dim=1))
        else:
            h, phi = self.mix_model(
                input_ids, attention_mask, mixup_indices=mixup_indices, lambda_=lambda_
            )
            return (
                self.classifier(masked_mean(h, attention_mask[:, :, None], dim=1)),
                phi,
            )

    def predict(self, input_ids, attention_mask):
        h = self.mix_model(input_ids, attention_mask)
        h = self.classifier(masked_mean(h, attention_mask[:, :, None], dim=1))  # [B, D]
        h = h / h.norm(dim=1, keepdim=True)
        l = self.label_matrix / self.label_matrix.norm(dim=1, keepdim=True)
        out = torch.mm(h, l.t())  # [B, C]
        pred = out.argmax(dim=1)
        return pred

    def get_label_embedding(self, labels):
        return self.label_matrix[labels]

    def load(self):
        self.mix_model.embedding_model.load()

    def get_embedding_model(self):
        return self.mix_model.embedding_model


class OoMMixSentenceClassificationModel(nn.Module):
    def __init__(self, mix_model, n_class):
        super().__init__()
        self.mix_model = mix_model
        self.classifier = create_sentence_classifier(
            mix_model.embedding_model.embed_dim, n_class
        )
        self.sentence_h = nn.Identity("sentence embedding")
        self.mix_sentence_h = nn.Identity("mixed sentence embedding")

    def forward(self, input_ids, attention_mask, mixup_indices=None, eps=None):
        if mixup_indices is None:
            h = self.mix_model(input_ids, attention_mask)
            h = self.sentence_h(masked_mean(h, attention_mask[:, :, None], dim=1))
            return self.classifier(h)
        else:
            h, mix_h, gamma, intr_loss = self.mix_model(
                input_ids, attention_mask, mixup_indices, eps
            )
            h = self.sentence_h(masked_mean(h, attention_mask[:, :, None], dim=1))
            mix_h = self.mix_sentence_h(
                masked_mean(mix_h, attention_mask[:, :, None], dim=1)
            )
            out = self.classifier(h)
            mix_out = self.classifier(mix_h)
            return out, mix_out, gamma, intr_loss

    def predict(self, input_ids, attention_mask):
        out = self.forward(input_ids, attention_mask)
        return out.argmax(dim=1)

    def load(self):
        self.mix_model.embedding_model.load()

    def get_embedding_model(self):
        return self.mix_model.embedding_model


def create_model(
    vocab_size=30522,
    embed_dim=768,
    padding_idx=0,
    drop_prob=0.1,
    n_head=12,
    k_dim=64,
    v_dim=64,
    feedforward_dim=3072,
    n_layer=12,
    augment="none",
    mixup_layer=3,
    max_length=256,
    d_class=16,
    d_layer=12,
    n_class=4,
):
    embedding_model = Bert(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=padding_idx,
        drop_prob=drop_prob,
        n_head=n_head,
        k_dim=k_dim,
        v_dim=v_dim,
        feedforward_dim=feedforward_dim,
        n_layer=n_layer,
    )
    if augment == "none":
        model = SentenceClassificationModel(embedding_model, n_class)
    elif augment == "tmix":
        model = TMixSentenceClassificationModel(
            embedding_model, mixup_layer=mixup_layer, n_class=n_class
        )
    elif augment == "nonlinearmix":
        model = NonlinearMixSentenceClassificationModel(
            embedding_model,
            max_length=max_length,
            mixup_layer=mixup_layer,
            d_class=d_class,
            n_class=n_class,
        )
    elif augment == "mixuptransformer":
        model = MixupTransformerSentenceClassificationModel(
            embedding_model, n_class=n_class
        )
    elif augment == "oommix":
        embedding_model = OoMMix(embedding_model, g_layer=mixup_layer, d_layer=d_layer)
        model = OoMMixSentenceClassificationModel(embedding_model, n_class)
    else:
        raise AttributeError("Invalid augment")
    return model


if __name__ == "__main__":
    m = create_model()
    for name, _ in m.named_modules():
        print(name)
