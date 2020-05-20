from pytorch_transformers import GPT2LMHeadModel
from torch.nn import CrossEntropyLoss


class GPT2ConditionalLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2ConditionalLMHeadModel, self).__init__(config)

    def forward(self, input_ids, position_ids=None, token_type_ids=None,
                labels=None, past=None, head_mask=None, reduction='mean'):
        transformer_outputs = self.transformer(
            input_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, past=past, head_mask=head_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction=reduction)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        # (loss), lm_logits, presents, (all hidden_states), (attentions)
        return outputs
