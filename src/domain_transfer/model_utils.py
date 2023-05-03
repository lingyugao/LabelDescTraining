# this file is modified from transformer 4.14.1
import math
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaConfig
import transformers.models.roberta.modeling_roberta as modeling_roberta


class RobertaClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Modified from huggingface transformer
    """
    def __init__(self, config, num_labels):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.config = config
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
        self.init_head()

    def init_head(self):
        """
        https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/models/roberta/modeling_roberta.py#L592
        """
        self.out_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.out_proj.bias.data.zero_()

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x


class TwoLayerClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Modified from huggingface transformer
    """
    def __init__(self, config, num_labels, params):
        super().__init__()
        self.config = config
        self.params = params
        self.dropout = self.dropout_func()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.nonlinear = self.activation_func()
        self.layer_norm = self.layernorm_func()
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

        self.init_head()

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    @staticmethod
    def pass_orig(x):
        return x

    def activation_func(self):
        if self.params.use_gelu:
            return self.gelu
        else:
            return torch.tanh

    def dropout_func(self):
        if self.params.add_dropout:
            classifier_dropout = (
                self.config.classifier_dropout if self.config.classifier_dropout is not None
                else self.config.hidden_dropout_prob
            )
            return nn.Dropout(classifier_dropout)
        else:
            return self.pass_orig

    def layernorm_func(self):
        if self.params.layer_norm:
            return nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        else:
            return self.pass_orig

    def init_head(self):
        """
        https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/models/roberta/modeling_roberta.py#L592
        """
        self.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.dense.bias.data.zero_()
        self.out_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.out_proj.bias.data.zero_()
        if self.params.layer_norm:
            self.layer_norm.bias.data.zero_()
            self.layer_norm.weight.data.fill_(1.0)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.nonlinear(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(modeling_roberta.RobertaPreTrainedModel):
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    Modified from huggingface transformer
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config_str, config, params, classifier_mask=False, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.config = config
        self.params = params
        self.classifier_mask = classifier_mask
        self.roberta = RobertaModel.from_pretrained(config_str, add_pooling_layer=False)
        if self.params.two_layer:
            self.lm_head = TwoLayerClassificationHead(config, num_labels, self.params)
        else:
            self.lm_head = RobertaClassificationHead(config, num_labels)

    @modeling_roberta.add_start_docstrings_to_model_forward(
        modeling_roberta.ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @modeling_roberta.add_code_sample_docstrings(
        processor_class=modeling_roberta._TOKENIZER_FOR_DOC,
        checkpoint=modeling_roberta._CHECKPOINT_FOR_DOC,
        output_type=modeling_roberta.SequenceClassifierOutput,
        config_class=modeling_roberta._CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.classifier_mask:
            if mask_pos < 0:
                mask_pos_1d = attention_mask.sum(axis=1) + mask_pos
                mask_pos_nd = mask_pos_1d.view(-1, 1, 1).expand(input_ids.shape[0], 1,
                                                             sequence_output.shape[2])
                output = sequence_output.gather(1, mask_pos_nd).view(input_ids.shape[0], -1)
                logits = self.lm_head(output)
            else:
                logits = self.lm_head(sequence_output[:, mask_pos, :])
        else:
            # take <s> token (equiv. to [CLS])
            logits = self.lm_head(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return modeling_roberta.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )