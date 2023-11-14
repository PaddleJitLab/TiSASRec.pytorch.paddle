import sys
sys.path.append('/home/aistudio/TiSASRec.pytorch.paddle/utils')
import paddle_aux
import paddle
import numpy as np
import sys
FLOAT_MIN = -sys.float_info.max


class PointWiseFeedForward(paddle.nn.Layer):

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = paddle.nn.Conv1D(in_channels=hidden_units,
            out_channels=hidden_units, kernel_size=1)
        self.dropout1 = paddle.nn.Dropout(p=dropout_rate)
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv1D(in_channels=hidden_units,
            out_channels=hidden_units, kernel_size=1)
        self.dropout2 = paddle.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        x = inputs
        perm_0 = list(range(x.ndim))
        perm_0[-1] = -2
        perm_0[-2] = -1
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.
            conv1(x.transpose(perm=perm_0))))))
        x = outputs
        perm_1 = list(range(x.ndim))
        perm_1[-1] = -2
        perm_1[-2] = -1
        outputs = x.transpose(perm=perm_1)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(paddle.nn.Layer):

    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = paddle.nn.Linear(in_features=hidden_size, out_features=
            hidden_size)
        self.K_w = paddle.nn.Linear(in_features=hidden_size, out_features=
            hidden_size)
        self.V_w = paddle.nn.Linear(in_features=hidden_size, out_features=
            hidden_size)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K,
        time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)
        Q_ = paddle.concat(x=paddle_aux.split(x=Q, num_or_sections=self.
            head_size, axis=2), axis=0)
        K_ = paddle.concat(x=paddle_aux.split(x=K, num_or_sections=self.
            head_size, axis=2), axis=0)
        V_ = paddle.concat(x=paddle_aux.split(x=V, num_or_sections=self.
            head_size, axis=2), axis=0)
        time_matrix_K_ = paddle.concat(x=paddle_aux.split(x=time_matrix_K,
            num_or_sections=self.head_size, axis=3), axis=0)
        time_matrix_V_ = paddle.concat(x=paddle_aux.split(x=time_matrix_V,
            num_or_sections=self.head_size, axis=3), axis=0)
        abs_pos_K_ = paddle.concat(x=paddle_aux.split(x=abs_pos_K,
            num_or_sections=self.head_size, axis=2), axis=0)
        abs_pos_V_ = paddle.concat(x=paddle_aux.split(x=abs_pos_V,
            num_or_sections=self.head_size, axis=2), axis=0)
        x = K_
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        attn_weights = Q_.matmul(y=paddle.transpose(x=x, perm=perm_2))
        x = abs_pos_K_
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        attn_weights += Q_.matmul(y=paddle.transpose(x=x, perm=perm_3))
        attn_weights += time_matrix_K_.matmul(y=Q_.unsqueeze(axis=-1)).squeeze(
            axis=-1)
        attn_weights = attn_weights / K_.shape[-1] ** 0.5
        time_mask = time_mask.unsqueeze(axis=-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(shape=[-1, -1, attn_weights.shape[-1]])
        attn_mask = attn_mask.unsqueeze(axis=0).expand(shape=[attn_weights.
            shape[0], -1, -1])
        paddings = paddle.ones(shape=attn_weights.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.dev)
        attn_weights = paddle.where(condition=time_mask, x=paddings, y=
            attn_weights)
        attn_weights = paddle.where(condition=attn_mask, x=paddings, y=
            attn_weights)
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)
        outputs = attn_weights.matmul(y=V_)
        outputs += attn_weights.matmul(y=abs_pos_V_)
        outputs += attn_weights.unsqueeze(axis=2).matmul(y=time_matrix_V_
            ).reshape(outputs.shape).squeeze(axis=2)
        outputs = paddle.concat(x=paddle_aux.split(x=outputs,
            num_or_sections=Q.shape[0], axis=0), axis=2)
        return outputs


class TiSASRec(paddle.nn.Layer):

    def __init__(self, user_num, item_num, time_num, args):
        super(TiSASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.item_emb = paddle.nn.Embedding(num_embeddings=self.item_num + 
            1, embedding_dim=args.hidden_units, padding_idx=0)
        self.item_emb_dropout = paddle.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb = paddle.nn.Embedding(num_embeddings=args.maxlen,
            embedding_dim=args.hidden_units)
        self.abs_pos_V_emb = paddle.nn.Embedding(num_embeddings=args.maxlen,
            embedding_dim=args.hidden_units)
        self.time_matrix_K_emb = paddle.nn.Embedding(num_embeddings=args.
            time_span + 1, embedding_dim=args.hidden_units)
        self.time_matrix_V_emb = paddle.nn.Embedding(num_embeddings=args.
            time_span + 1, embedding_dim=args.hidden_units)
        self.item_emb_dropout = paddle.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb_dropout = paddle.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = paddle.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = paddle.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = paddle.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = paddle.nn.LayerList()
        self.attention_layers = paddle.nn.LayerList()
        self.forward_layernorms = paddle.nn.LayerList()
        self.forward_layers = paddle.nn.LayerList()
        self.last_layernorm = paddle.nn.LayerNorm(normalized_shape=args.
            hidden_units, epsilon=1e-08)
        for _ in range(args.num_blocks):
            new_attn_layernorm = paddle.nn.LayerNorm(normalized_shape=args.
                hidden_units, epsilon=1e-08)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units,
                args.num_heads, args.dropout_rate, args.device)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = paddle.nn.LayerNorm(normalized_shape=args.
                hidden_units, epsilon=1e-08)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.
                dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        seqs = self.item_emb(paddle.to_tensor(data=log_seqs, dtype='int64')
            .to(self.dev))
        seqs *= self.item_emb._embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.
            shape[0], 1])
        positions = paddle.to_tensor(data=positions, dtype='int64').to(self.dev
            )
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)
        time_matrices = paddle.to_tensor(data=time_matrices, dtype='int64').to(
            self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)
        timeline_mask = paddle.to_tensor(data=log_seqs == 0, dtype='bool').to(
            self.dev)
        seqs *= ~timeline_mask.unsqueeze(axis=-1)
        tl = seqs.shape[1]
        attention_mask = ~paddle.tril(x=paddle.ones(shape=(tl, tl), dtype=
            'bool'))
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs, timeline_mask,
                attention_mask, time_matrix_K, time_matrix_V, abs_pos_K,
                abs_pos_V)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(axis=-1)
        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs):
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)
        pos_embs = self.item_emb(paddle.to_tensor(data=pos_seqs, dtype=
            'int64').to(self.dev))
        neg_embs = self.item_emb(paddle.to_tensor(data=neg_seqs, dtype=
            'int64').to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(axis=-1)
        neg_logits = (log_feats * neg_embs).sum(axis=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, time_matrices, item_indices):
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)
        final_feat = log_feats[:, (-1), :]
        item_embs = self.item_emb(paddle.to_tensor(data=item_indices, dtype
            ='int64').to(self.dev))
        logits = item_embs.matmul(y=final_feat.unsqueeze(axis=-1)).squeeze(axis
            =-1)
        return logits
