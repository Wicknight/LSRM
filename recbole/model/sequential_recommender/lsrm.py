import torch
from torch import nn
import torch.nn.functional as fn
from torch.distributions.normal import Normal
import numpy as np
import copy

from typing import Optional, Tuple, Union, List
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MoEEmbedExperts, VanillaAttention, SparseDispatcher
from recbole.model.loss import BPRLoss

from transformers.configuration_utils import PretrainedConfig
# from transformers import BertEncoder
#from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Block, BaseModelOutputWithPastAndCrossAttentions


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        # embedding dropout is the same
        self.drop = nn.Dropout(config.embd_pdrop)

        # different from standard transformers, where we use decreased dropout rates for layers
        resid_dropout_list = config.resid_pdrop
        attn_dropout_list = config.attn_drop
        #print('resid dropout list', resid_dropout_list)
        #print('attn dropout list', attn_dropout_list)
        module_list = []
        for i in range(config.num_hidden_layers):
            config.resid_pdrop = resid_dropout_list[i]
            config.attn_drop = attn_dropout_list[i]
            module = GPT2Block(config, layer_idx=i)
            module_list.append(module)
        self.h = nn.ModuleList(module_list)

        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "`GPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def generate(self):
        pass


class TransformerModel(nn.Module):

    def __init__(
        self,
        model_type='GPT',
        tr_config=None,
        vocab_size=None,
        init_pretrained=False,
    ):
        super().__init__()

        self.model_type = model_type

        if model_type == "GPT":
            config = GPT2Config(
                vocab_size=tr_config["vocab_size"],
                n_positions=tr_config["max_seq_length"],
                max_position_embeddings=tr_config["max_seq_length"],
                n_embd=tr_config["hidden_size"],
                n_layer=tr_config["n_layers"],
                n_head=tr_config["n_heads"],
                n_inner=tr_config["inner_size"],
                activation_function=tr_config["hidden_act"],
                resid_pdrop=tr_config["resid_pdrop"],
                embd_drop=tr_config["embd_drop"],
                attn_drop=tr_config["attn_pdrop"],
                layer_norm_epsilon=tr_config["layer_norm_eps"],
                initializer_range=tr_config["initializer_range"],
                bos_token_id=tr_config["vocab_size"],
                eos_token_id=tr_config["vocab_size"],
            )
        elif model_type == "BERT":
            config = BERTConfig(
                vocab_size=tr_config["vocab_size"],
                max_position_embeddings=tr_config["max_seq_length"],
                hidden_size=tr_config["hidden_size"],
                num_hidden_layers=tr_config["n_layers"],
                num_attention_heads=tr_config["n_heads"],
                intermediate_size=tr_config["inner_size"],
                hidden_act=tr_config["hidden_act"],
                hidden_dropout_prob=tr_config["embd_drop"],
                attention_probs_dropout_prob=tr_config["attn_pdrop"],
                layer_norm_epsilon=tr_config["layer_norm_eps"],
                initializer_range=tr_config["initializer_range"],
                pad_token_id=0,
            )

        #config.is_decoder = True
        #config.add_cross_attention = True

        if init_pretrained:
            
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            del temp_bert.embeddings
            del temp_bert.pooler
            self.input_transformers = temp_bert.encoder
            print('initializing from pretrained bert.')
        else:
            #print(config)
            if model_type == 'GPT':
                self.input_transformers = GPT2Model(config)
                #print('input transformers', self.input_transformers)
            elif model_type == 'BERT':
                self.input_transformers = BertModel(config)

        #self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        #self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    # def convert_to_fp16(self):
    #     """
    #     Convert the torso of the model to float16.
    #     """
    #     self.input_blocks.apply(convert_module_to_f16)
    #     self.middle_block.apply(convert_module_to_f16)
    #     self.output_blocks.apply(convert_module_to_f16)
    #
    # def convert_to_fp32(self):
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.input_blocks.apply(convert_module_to_f32)
    #     self.middle_block.apply(convert_module_to_f32)
    #     self.output_blocks.apply(convert_module_to_f32)
    #

    def parallelize_inference(self):
        from parallelformers import parallelize
        parallelize(self.input_transformers, num_gpus=2, fp16=True, verbose=None)
        print('transformer', self.input_transformers)
        assert False
    
    def parallelize(self):
        pass

    def get_embeds(self, input_ids):
        if self.model_type == 'GPT':
            return self.input_transformers.wte(input_ids)
        elif self.model_type == 'BERT':
            return self.input_transformers.embeddings.word_embeddings(input_ids)

    def forward(self, input_ids, position_ids, inputs_embeds=None, return_last=True):
        all_outputs = self.input_transformers(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        outputs = all_outputs
        if return_last:
            return outputs.last_hidden_state
        else:
            return outputs

    def generate(self, input_ids, max_length):
        return self.input_transformers.generate(input_ids, max_length)


class LSRM(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(LSRM, self).__init__(config, dataset)

        # load parameters info
        model_type = config["model_type"]
        self.loss_type = config["loss_type"]
        config["max_seq_length"] = self.max_seq_length
        self.initializer_range = config["initializer_range"]
        self.is_seq_list = config["is_seq_list"]
            
        resid_pdrop_start = config["resid_pdrop_start"]
        resid_pdrop_end = config["resid_pdrop_end"]
        attn_pdrop_start = config["attn_pdrop_start"]
        attn_pdrop_end = config["attn_pdrop_end"]
        config["resid_pdrop"] = np.linspace(resid_pdrop_start, resid_pdrop_end, config["n_layers"])
        config["attn_pdrop"] = np.linspace(attn_pdrop_start, attn_pdrop_end, config["n_layers"])
        config["vocab_size"] = self.n_items

        self.task = config["recommendation_task"]
        self.data_scaling = config["data_scaling"]
        if self.task == 'robust':
            self.NOISY_ITEM_SEQ = config["NOISY_ITEM_SEQ"]
        if self.data_scaling:
            self.SPARSE_ITEM_SEQ = config["SPARSE_ITEM_SEQ"]
            self.SPARSE_ITEM_SEQ_LEN = config["SPARSE_ITEM_SEQ_LEN"]

        self.transformer = TransformerModel(
                model_type=model_type,
                tr_config=config,
                vocab_size=self.n_items, # +2
                init_pretrained=False,
            )
        
        self.parallel = config["parallel"]

        if self.parallel:
            self.parallelize()
        #assert False

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    def parallelize(self):
        self.transformer.parallelize()
    
    def get_code_embeddings(self, item_ids):
        item_codes = self.item_codes[item_ids]
        item_code_embeddings = self.item_code_embeddings(item_codes)

        process_embeddings = torch.mean(item_code_embeddings, dim=-2, keepdim=False)

        return process_embeddings

    def group_users(self, user_tokens, dataset):
        user_list = []
        for user_token in user_tokens:
            user_list.append(dataset.field2token_id['user_id'][user_token])
        return user_list

    def forward(self, item_seq, item_seq_len):
        #item_seq = item_seq.to(self.first_device)
        #item_seq_len = item_seq_len.to(self.first_device)
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        extended_attention_mask = self.get_attention_mask(item_seq)

        outputs = self.transformer(item_seq, position_ids, return_last=True)
        output = self.gather_indexes(outputs, item_seq_len - 1)

        return output

    def calculate_loss(self, interaction):
        self.test_loss = 0.0
        self.test_num = 0
        if self.data_scaling:
            item_seq = interaction[self.SPARSE_ITEM_SEQ]
            item_seq_len = interaction[self.SPARSE_ITEM_SEQ_LEN]
            #print('sparse item seq', item_seq)
            #print('sparse item seq len', item_seq_len)
        else:
            item_seq = interaction[self.ITEM_SEQ]
            #print('batch size', item_seq.shape[0])
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]

            pos_items_emb = self.transformer.get_embeds(pos_items)
            neg_items_emb = self.transformer.get_embeds(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)

            return loss
        else:  # self.loss_type = 'CE'
            test_item_ids = torch.arange(start=0, end=self.n_items, device=self.device)
            test_item_emb = self.transformer.get_embeds(test_item_ids)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            #print('loss', loss)

            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID].to(self.device)
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.transformer.get_embeds(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        if self.task == 'robust':
            item_seq = interaction[self.NOISY_ITEM_SEQ]
        else:
            item_seq = interaction[self.ITEM_SEQ]
        
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        #print('item seq', item_seq)
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_ids = torch.arange(start=0, end=self.n_items, device=self.device)
        test_items_emb = self.transformer.get_embeds(test_items_ids)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        return scores

    def seq_topk_predict(self, src_data, tgt_data):
        
        hr, ndcg = {}, {}
        hr[0] = 0.0
        ndcg[0] = 0.0
        test_items_ids = torch.arange(start=0, end=self.n_items, device=self.device)
        test_items_emb = self.transformer.get_embeds(test_items_ids)
        for i in range(10):
            hr[i+1] = 0.0
        pred_tgt_data = []
        for (src_sample, tgt_sample) in zip(src_data, tgt_data)
            seq_len = len(src_sample)
            sample_data = copy.deepcopy(src_sample)
            sample = torch.LongTensor(np.array(sample_data))
            pre_sample = sample.to(self.device)
            sample = pre_sample.unsqueeze(0)
            sample_len = torch.tensor([seq_len], dtype=torch.long, device=self.device).unsqueeze(0)
            #print('sample', sample)
            #print('sample len', sample_len)
            seq_output = self.forward(sample, sample_len)
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
            item = self.decode(scores)[0]
            #print(item)
            for j in range(10):
                hr_nums[j+1] = hr_nums[j] + 1 if item[j].item() in tgt_sample[:j+1] else hr_nums[j]
                ndcg[j+1] = self.get_tr(sample_data[-(j+1):], tgt_sample[:j+1])
            pred_tgt_data.append(sample_data[-10:])
            #print('hr nums', hr_nums)
            for j in range(10):
                hr[j+1] += hr_nums[j+1] / 10

        for j in range(10):
            hr[j+1] /= len(tgt_data)
            ndcg[j+1] /= len(tgt_data)
        bleu = self.cal_bleu(tgt_data, pred_tgt_data)

        return hr, bleu, ndcg

    def cal_bleu(self, golden_data, pred_data):
        from bleu_mp import compute_bleu
        for i in range(len(golden_data)):
            golden_data[i] = [golden_data[i]]
        return compute_bleu(pred_data, golden_data)

    def get_dtr(self, scores):
        return np.sum(
            np.divide(
                np.power(2, scores) - 1,
                np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)
            ), dtype=np.float32
        )

    def get_tr(self, rank_list, pos_items):
        relevance = np.ones_like(pos_items)
        it2rel = {it: r for it, r in zip(pos_items, relevance)}
        rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

        idtr = cls.get_dtr(relevance)

        dtr = cls.get_dtr(rank_scores)

        if dcg == 0.0:
            return 0.0

        tr = dtr / idtr
        return tr
    
    def decode(self, scores):
        values, indices = scores.topk(10, dim=1, largest=True)
        #print('indices shape', indices.shape)
        #print(indices)
        return indices