"""Pretrained ViT attention feedback at logical KV page granularity."""
from __future__ import annotations

import time
from types import MethodType


class ViTAttentionFeedback:
    """Measure class-token attention to visual patches from a pretrained ViT."""

    MODEL_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"

    def __init__(self, page_tokens: int, device: str = "cpu", kv_reuse: bool = False, reuse_window: int = 8):
        import timm
        import torch

        if page_tokens <= 0 or reuse_window <= 0:
            raise ValueError("page_tokens and reuse_window must be positive")
        self.kv_reuse = kv_reuse
        self.reuse_window = reuse_window
        self._reuse_inputs = {}
        self._reuse_generation = None
        self._reuse_owner = None
        self._reuse_signature = None
        self.reuse_stats = None
        self.torch = torch
        self.page_tokens = page_tokens
        self.device = torch.device(device)
        self.model = timm.create_model(self.MODEL_NAME, pretrained=True).to(self.device).eval()
        self.model.blocks[-1].attn.fused_attn = False
        config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**config, is_training=False)
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.input_size = config["input_size"][-1]
        self._attention = None
        self._qkv = None
        self.last_inference_seconds = 0.0
        self.baseline_comparison = None
        self.kv_consumed_pages = 0
        self.model.blocks[-1].attn.attn_drop.register_forward_hook(self._capture_attention)
        self.model.blocks[-1].attn.qkv.register_forward_hook(self._capture_qkv)

    def source_crop(self, source_size):
        """Invert the selected timm Resize/CenterCrop geometry exactly."""
        from torchvision.transforms import Resize, CenterCrop
        from torchvision.transforms.functional import _compute_resized_output_size

        width, height = source_size
        current_width, current_height = width, height
        left = top = 0.0
        scale_x = scale_y = 1.0
        for transform in self.transform.transforms:
            if isinstance(transform, Resize):
                size = [transform.size] if isinstance(transform.size, int) else list(transform.size)
                new_height, new_width = _compute_resized_output_size(
                    (current_height, current_width), size, transform.max_size)
                scale_x *= new_width / current_width
                scale_y *= new_height / current_height
                current_width, current_height = new_width, new_height
            elif isinstance(transform, CenterCrop):
                crop_height, crop_width = transform.size
                if crop_width > current_width or crop_height > current_height:
                    raise ValueError("model preprocessing with crop padding is not supported")
                left += round((current_width - crop_width) / 2) / scale_x
                top += round((current_height - crop_height) / 2) / scale_y
                current_width, current_height = crop_width, crop_height
        if (current_width, current_height) != (self.input_size, self.input_size):
            raise ValueError("unsupported model preprocessing geometry")
        return (left, top, self.input_size / scale_x, self.input_size / scale_y)

    def _capture_attention(self, _module, _inputs, output) -> None:
        self._attention = output.detach()

    def _capture_qkv(self, _module, _inputs, output) -> None:
        self._qkv = output.detach()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def page_scores(self, rgb_frame, controller=None, generation: int | None = None,
                    compare_baseline: bool = False) -> dict[int, float]:
        from PIL import Image

        image = Image.fromarray(rgb_frame)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        self._attention = None
        self._qkv = None
        self.baseline_comparison = None
        self.kv_consumed_pages = 0
        self.reuse_stats = None
        if controller is not None and generation is None:
            raise ValueError("generation is required for KV-backed inference")
        if self.kv_reuse and controller is None:
            raise ValueError("cross-frame reuse requires KV-backed inference")
        baseline_logits = None
        if compare_baseline:
            self._synchronize()
            baseline_start = time.perf_counter()
            with self.torch.inference_mode():
                baseline_logits = self.model(tensor)
            self._synchronize()
            baseline_seconds = time.perf_counter() - baseline_start
            self._attention = None
            self._qkv = None
        attention = self.model.blocks[-1].attn
        original_forward = attention.forward
        if controller is not None:
            attention.forward = MethodType(
                lambda module, tokens, attn_mask=None, is_causal=False:
                self._stored_attention(module, tokens, controller, generation, attn_mask, is_causal),
                attention,
            )
        self._synchronize()
        start = time.perf_counter()
        try:
            with self.torch.inference_mode():
                logits = self.model(tensor)
            self._synchronize()
        except Exception:
            self._reuse_inputs.clear()
            self._reuse_generation = None
            self._reuse_owner = None
            raise
        finally:
            attention.forward = original_forward
        self.last_inference_seconds = time.perf_counter() - start
        if baseline_logits is not None:
            self.baseline_comparison = {
                "baseline_seconds": baseline_seconds,
                "same_top1": bool(baseline_logits.argmax() == logits.argmax()),
                "mean_abs_logit_error": float((baseline_logits - logits).abs().mean()),
            }
        if self._attention is None:
            raise RuntimeError("ViT attention hook did not run")
        prefix = self.model.num_prefix_tokens
        patch_attention = self._attention[0, :, 0, prefix:].mean(dim=0).float()
        peak = patch_attention.max().clamp_min(1e-12)
        normalized = (patch_attention / peak).cpu().tolist()
        scores: dict[int, float] = {}
        for token, score in enumerate(normalized):
            page = token // self.page_tokens
            scores[page] = max(scores.get(page, 0.0), score)
        return scores

    def _stored_attention(self, attention, tokens, controller, generation, attn_mask, is_causal):
        if attn_mask is not None or is_causal:
            raise ValueError("KV-backed ViT attention does not support masks")
        batch, sequence, _ = tokens.shape
        if batch != 1:
            raise ValueError("KV-backed ViT attention supports one frame at a time")
        gate = attention.gate(tokens).sigmoid() if attention.gate is not None else None
        prefix = self.model.num_prefix_tokens
        if controller.page_count != (sequence - prefix + self.page_tokens - 1) // self.page_tokens:
            raise ValueError("controller and model patch-page geometry differ")
        if self.kv_reuse:
            query, key, value = self._reuse_attention_kv(attention, tokens, controller, generation)
        else:
            qkv = attention.qkv(tokens).reshape(batch, sequence, 3, attention.num_heads, attention.head_dim)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            query, key = attention.q_norm(query), attention.k_norm(key)
        stored_keys = [key[:, :, :prefix]]
        stored_values = [value[:, :, :prefix]]
        for page, offset in enumerate(range(prefix, sequence, self.page_tokens)):
            end = min(offset + self.page_tokens, sequence)
            if not self.kv_reuse:
                controller.attach_kv(page, generation, key[0, :, offset:end], value[0, :, offset:end],
                                     device=str(self.device))
            stored_key, stored_value = controller.read_kv(page, generation)
            stored_keys.append(stored_key.unsqueeze(0).to(device=query.device, dtype=query.dtype))
            stored_values.append(stored_value.unsqueeze(0).to(device=query.device, dtype=query.dtype))
            self.kv_consumed_pages += 1
        key = self.torch.cat(stored_keys, dim=2)
        value = self.torch.cat(stored_values, dim=2)
        weights = (query * attention.scale) @ key.transpose(-2, -1)
        weights = attention.attn_drop(weights.softmax(dim=-1))
        output = (weights @ value).transpose(1, 2).reshape(batch, sequence, attention.attn_dim)
        output = attention.norm(output)
        if gate is not None:
            output = output * gate
        return attention.proj_drop(attention.proj(output))

    def _reuse_attention_kv(self, attention, tokens, controller, generation):
        """Project Q for all tokens, K/V only for prefix tokens and misses.

        Codec eligibility alone is insufficient: upstream attention is global.
        Exact final-block input equality is mandatory for each reusable page.
        """
        torch = self.torch
        batch, sequence, _ = tokens.shape
        prefix = self.model.num_prefix_tokens
        dimension = attention.num_heads * attention.head_dim
        weight, bias = attention.qkv.weight, attention.qkv.bias
        if attention.training:
            raise ValueError("cross-frame KV reuse requires evaluation mode")
        if weight.shape[0] != 3 * dimension:
            raise ValueError("cross-frame reuse requires standard packed Q/K/V projection")
        signature = (tuple(tokens.shape), tokens.dtype, tokens.device,
                     tuple((name, id(parameter), parameter._version) for name, parameter in attention.named_parameters()))
        previous = self._reuse_generation
        valid_history = (self._reuse_owner is controller and previous is not None
                         and generation == previous + 1 and self._reuse_signature == signature)
        query = torch.nn.functional.linear(tokens, weight[:dimension], None if bias is None else bias[:dimension])
        query = attention.q_norm(query.reshape(batch, sequence, attention.num_heads, attention.head_dim).transpose(1, 2))
        spans = [(page, start, min(start + self.page_tokens, sequence))
                 for page, start in enumerate(range(prefix, sequence, self.page_tokens))]
        hits, misses, reasons = {}, [], {}
        for page, start, end in spans:
            stored = controller.tensor_store.records.get(page) if controller.tensor_store is not None else None
            old_input = self._reuse_inputs.get(page)
            reason = None
            if not valid_history or stored is None or stored.generation != previous:
                reason = "no_valid_history"
            elif page not in controller.reusable_pages:
                reason = "codec_or_temporal_invalidation"
            elif stored.origin_generation is None or generation - stored.origin_generation > self.reuse_window:
                reason = "window_expired"
            elif old_input is None or not torch.equal(old_input, tokens[:, start:end]):
                reason = "model_input_changed"
            if reason is None:
                hits[page] = {"page_id": page, "source_generation": previous,
                              "origin_generation": stored.origin_generation}
            else:
                misses.append((page, start, end))
                reasons[reason] = reasons.get(reason, 0) + 1
        pieces = [tokens[:, :prefix]] + [tokens[:, start:end] for _, start, end in misses]
        packed = torch.cat(pieces, dim=1)
        projected = torch.nn.functional.linear(packed, weight[dimension:], None if bias is None else bias[dimension:])
        kv = projected.reshape(batch, packed.shape[1], 2, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
        key, value = attention.k_norm(kv[0]), kv[1]
        cursor = prefix
        for page, start, end in misses:
            count = end - start
            controller.attach_kv(page, generation, key[0, :, cursor:cursor + count],
                                 value[0, :, cursor:cursor + count], device=str(self.device))
            cursor += count
        for page in hits:
            controller.reuse_kv(page, generation, previous)
        self._reuse_inputs = {page: tokens[:, start:end].detach().clone() for page, start, end in spans}
        self._reuse_generation = generation
        self._reuse_owner = controller
        self._reuse_signature = signature
        self.reuse_stats = {
            "enabled": True, "window": self.reuse_window,
            "pages_reused": len(hits), "pages_recomputed": len(misses),
            "tokens_kv_reused": sum(end - start for page, start, end in spans if page in hits),
            "tokens_kv_projected": packed.shape[1],
            "validation_input_bytes": sum(item.numel() * item.element_size() for item in self._reuse_inputs.values()),
            "reused_pages": list(hits.values()), "recompute_reasons": reasons,
        }
        # The common attention path reads every patch page from the store;
        # only prefix K/V is needed directly from this projection.
        return query, key[:, :, :prefix], value[:, :, :prefix]

    def page_kv(self) -> dict[int, tuple[object, object]]:
        """Return last-block patch K/V tensors grouped by controller page."""
        if self._qkv is None:
            raise RuntimeError("run page_scores before reading model KV tensors")
        attention = self.model.blocks[-1].attn
        batch, sequence, combined = self._qkv.shape
        if batch != 1:
            raise ValueError("only one decoded frame is supported per inference")
        heads = attention.num_heads
        head_dim = combined // (3 * heads)
        qkv = self._qkv.reshape(batch, sequence, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        prefix = self.model.num_prefix_tokens
        key = qkv[1, 0, :, prefix:, :]
        value = qkv[2, 0, :, prefix:, :]
        pages = {}
        for page, start in enumerate(range(0, key.shape[1], self.page_tokens)):
            pages[page] = (
                key[:, start:start + self.page_tokens].contiguous(),
                value[:, start:start + self.page_tokens].contiguous(),
            )
        return pages
