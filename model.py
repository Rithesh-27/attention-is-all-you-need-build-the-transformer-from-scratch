"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    dictionary = {}
    for idx,ele in enumerate(specials):
        dictionary[ele] = idx
    curr_id = len(specials)
    for sentence in sentences:
        for word in sentence.split():
            if word not in dictionary:
                dictionary[word] = curr_id
                curr_id += 1
    return dictionary

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    inv_dictionary = {}
    for token, num_id in token_to_id.items():
        inv_dictionary[num_id] = token
    return inv_dictionary

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    encoded_sentence = []
    for word in sentence.split():
        encoded_sentence.append(token_to_id.get(word, token_to_id[unk_token]))
    return encoded_sentence

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    decoded_sentence = []
    for num_id in ids:
        decoded_sentence.append(id_to_token[num_id])
    return decoded_sentence

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    padded_sequence = [pad_id for _ in range(max_len)]
    for i in range(max_len):
        if i >= len(ids):
            break
        padded_sequence[i] = ids[i]
    return padded_sequence

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    return torch.tensor(padded_sequences, dtype = torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    return math.sqrt(d_model) * embeddings

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    div_vector = []
    for i in range(d_model // 2):
        div_vector.append(10000 ** ((-2 * i) / d_model))
    return torch.tensor(div_vector, dtype = torch.float)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    position_idx = []
    for i in range(max_len):
        position_idx.append([i])
    return torch.tensor(position_idx, dtype=torch.float)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    for i,term in enumerate(div_term.tolist()):
        pe[:,2*i] = torch.sin(position.squeeze(1) * term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    for i, term in enumerate(div_term.tolist()):
        pe[:,2*i+1] = torch.cos(position.squeeze(1) * term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    positional_encodings = torch.zeros(max_len,d_model)
    div_term = compute_positional_div_term(d_model)
    position_index = build_position_index_column(max_len)

    fill_even_indices_with_sin(positional_encodings, position_index, div_term)
    fill_odd_indices_with_cos(positional_encodings, position_index, div_term)

    return positional_encodings

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    B,L,d_model = embedded_batch.shape
    input_tensor = embedded_batch[:,:,:] + positional_encoding[:L,:]
    return input_tensor

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    mask = token_ids != pad_id
    mask = mask.unsqueeze(1).unsqueeze(1)
    return mask

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    causal_mask = torch.randn(seq_len, seq_len)
    causal_mask = torch.tril(causal_mask)
    causal_mask = torch.tensor(causal_mask, dtype=torch.bool)
    return causal_mask.reshape(1,1,seq_len,seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    combined_mask = torch.logical_and(padding_mask, causal_mask)
    return combined_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    raw_attention_scores = torch.matmul(query, torch.transpose(key, -1,-2))
    return raw_attention_scores

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return scores.masked_fill(~mask, float("-inf"))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    weights = torch.nn.functional.softmax(masked_scores, dim=-1)
    return torch.nan_to_num(weights, nan=0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    d_k = query.shape[-1]
    raw_attention_scores = compute_raw_attention_scores(query, key)
    attention_scores = scale_attention_scores(raw_attention_scores, d_k)
    if mask is not None:
        attention_scores = mask_attention_scores_with_neg_inf(attention_scores, mask)
    attention_weights = softmax_attention_weights(attention_scores)
    context = apply_attention_weights_to_values(attention_weights, value)

    return (context, attention_weights)

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    B,L,D = tensor.shape
    return tensor.reshape(B,L, num_heads, D // num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    return split_tensor.permute(0,2,1,3)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    multi_head_tensor = multi_head_tensor.transpose(1,2)
    multi_head_tensor = multi_head_tensor.contiguous()
    B,L,H,d = multi_head_tensor.shape
    return multi_head_tensor.reshape(B,L,H*d)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    if bias is None:
        return x @ weight.T
    return x @ weight.T + bias

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    Q = apply_linear_projection(x, w_q, b_q)
    K = apply_linear_projection(x, w_k, b_k)
    V = apply_linear_projection(x, w_v, b_v)

    return (Q,K,V)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))

    return (q_h, k_h, v_h)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    merged_heads = merge_heads_back_to_model_dim(context)
    multi_head_block_output = apply_linear_projection(merged_heads, w_o, b_o)
    return multi_head_block_output

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    Q = project_to_query_key_value(query, w_q, None, w_k, None, w_v, None)[0]
    K = project_to_query_key_value(key, w_q, None, w_k, None, w_v, None)[1]
    V = project_to_query_key_value(value, w_q, None, w_k, None, w_v, None)[2]
    Q,K,V = split_qkv_into_heads(Q, K, V, num_heads)
    context, weights = multi_head_scaled_dot_product_attention(Q,K,V,mask)
    output = merge_heads_and_project_output(context, w_o, None)
    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    y = x @ w1 + b1
    return torch.relu(y)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    y = hidden @ w2 + b2
    return y

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x,w1,b1),w2,b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    mean = torch.mean(x, -1, keepdim=True)
    variance = torch.var(x, -1, keepdim=True, correction=0)
    return (mean,variance)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, variance = compute_layer_norm_mean_and_variance(x)
    x_c = (x - mean) / torch.sqrt(variance + eps)
    return gamma * x_c + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    return normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
   return (x * keep_mask) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    attention_output = assemble_multi_head_attention_forward(x,x,x,w_q,w_k,w_v,w_o,num_heads,src_mask)
    final_output = apply_residual_add_and_norm(x,attention_output,gamma,beta)
    return final_output

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    ffn_output = position_wise_feed_forward_network(x,w1,b1,w2,b2)
    final_output = apply_residual_add_and_norm(x,ffn_output,gamma,beta)
    return final_output

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    w_q = layer_params["w_q"]
    w_k = layer_params["w_k"]
    w_v = layer_params["w_v"]
    w_o = layer_params["w_o"]
    attn_gamma = layer_params["attn_gamma"]
    attn_beta = layer_params["attn_beta"]

    w1 = layer_params["w1"]
    b1 = layer_params["b1"]
    w2 = layer_params["w2"]
    b2 = layer_params["b2"]
    ffn_gamma = layer_params["ffn_gamma"]
    ffn_beta = layer_params["ffn_beta"]

    attn_output = encoder_layer_self_attention_sublayer(x,w_q,w_k,w_v,w_o,attn_gamma,attn_beta,num_heads,src_mask)
    ffn_output = encoder_layer_feed_forward_sublayer(attn_output,w1,b1,w2,b2,ffn_gamma,ffn_beta)
    return ffn_output

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    curr_input = x
    for params in encoder_layer_params_list:
        output = assemble_encoder_layer(curr_input,params,num_heads,src_mask)
        curr_input = output
    final_output = curr_input
    return final_output

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    attn_output = assemble_multi_head_attention_forward(y,y,y,w_q,w_k,w_v,w_o,num_heads,tgt_mask)
    final_output = apply_residual_add_and_norm(y,attn_output,gamma,beta)
    return final_output

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(1).unsqueeze(1)
    attn_output = assemble_multi_head_attention_forward(y,encoder_output,encoder_output,w_q,w_k,w_v,w_o,num_heads,src_mask)
    final_output = apply_residual_add_and_norm(y,attn_output,gamma,beta)
    return final_output

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    ffn_output = position_wise_feed_forward_network(y,w1,b1,w2,b2)
    final_output = apply_residual_add_and_norm(y,ffn_output,gamma,beta)
    return final_output

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    self_attn_output = decoder_layer_masked_self_attention_sublayer(y,layer_params["w_q_self"],layer_params["w_k_self"],layer_params["w_v_self"],layer_params["w_o_self"],layer_params["self_gamma"],layer_params["self_beta"],num_heads,tgt_mask)
    cross_attn_output = decoder_layer_cross_attention_sublayer(self_attn_output,encoder_output,layer_params["w_q_cross"],layer_params["w_k_cross"],layer_params["w_v_cross"],layer_params["w_o_cross"],layer_params["cross_gamma"],layer_params["cross_beta"],num_heads,src_mask)
    final_output = decoder_layer_feed_forward_sublayer(cross_attn_output,layer_params["w1"],layer_params["b1"],layer_params["w2"],layer_params["b2"],layer_params["ffn_gamma"],layer_params["ffn_beta"])
    return final_output

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    curr_input = y
    for params in decoder_layer_params_list:
        output = assemble_decoder_layer(curr_input,encoder_output,params,num_heads,src_mask,tgt_mask)
        curr_input = output
    final_output = curr_input
    return final_output

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    return apply_linear_projection(decoder_output,output_projection_weight,output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    return token_embedding_weight.transpose(0,1)

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    return torch.nn.functional.log_softmax(logits,dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    token_embeddings = model_params["token_embedding"]
    output_projection = model_params["output_projection"]
    d_model = token_embeddings.shape[1]
    max_len = max(src_ids.shape[1],tgt_ids.shape[1])
    src_len = src_ids.shape[1]
    tgt_len = tgt_ids.shape[1]

    # Embed the tokens
    src_embeddings = scale_embeddings_by_sqrt_d_model(token_embeddings[src_ids],d_model)
    tgt_embeddings = scale_embeddings_by_sqrt_d_model(token_embeddings[tgt_ids],d_model)

    # Build and add positional encodings
    positional_encoding = build_sinusoidal_positional_encoding(max_len,d_model)
    src_embeddings_encoded = add_positional_encoding_to_embeddings(src_embeddings,positional_encoding)
    tgt_embeddings_encoded = add_positional_encoding_to_embeddings(tgt_embeddings,positional_encoding)

    # Build src and tgt masks
    src_mask = combine_padding_and_causal_masks(build_padding_mask(src_ids,pad_id),build_causal_mask(src_len))
    tgt_mask = combine_padding_and_causal_masks(build_padding_mask(tgt_ids,pad_id),build_causal_mask(tgt_len))

    # Encoder and Decoder forward prop
    encoder_layers_params = model_params["encoder_layers"]
    decoder_layers_params = model_params["decoder_layers"]

    encoder_output = stack_encoder_layers(src_embeddings_encoded,encoder_layers_params,num_heads,src_mask)
    decoder_output = stack_decoder_layers(tgt_embeddings_encoded,encoder_output,decoder_layers_params,num_heads,src_mask,tgt_mask)

    final_logits = apply_log_softmax_over_vocab(apply_final_output_projection(decoder_output,output_projection))
    return final_logits

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

