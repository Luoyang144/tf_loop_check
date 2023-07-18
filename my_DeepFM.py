import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.python.framework import sparse_tensor

class DeepFM(tf.keras.Model):
    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[0., 0.],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 l2_reg=0.001, zscore_file=None,
                 learning_rate=0.001, training=False, use_bn=False,
                 multi_fea=[], multihot_fea=[], preemb_fea=[]):
        # print(locals())
        super(DeepFM, self).__init__()
        self.is_save_model = False
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.training = training
        self.multi_fea = multi_fea
        self.multihot_fea = multihot_fea
        self.multi_hot_fea_tf = tf.constant(self.multihot_fea, dtype=tf.int64)
        self.len_multihot_fea = tf.constant(len(self.multihot_fea), dtype=tf.int64)
        self.preemb_fea = preemb_fea
        self.field_size_single = field_size - len(multihot_fea) - len(preemb_fea)

        for x in self.multi_fea:
            self.field_size_single += (x[1] - x[0] - 1)

        self.loss = tf.keras.losses.binary_crossentropy
        self.lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, decay_steps=1000000, decay_rate=1, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')

        zscore_mean_np, zscore_var_np = self.zscore_np(feature_size, zscore_file)
        self.zscore_mean = tf.keras.layers.Embedding(feature_size, 1, weights=[zscore_mean_np], trainable=False)
        self.zscore_var = tf.keras.layers.Embedding(feature_size, 1, weights=[zscore_var_np], trainable=False)
        self.emb_fm = tf.keras.layers.Embedding(feature_size, embedding_size, embeddings_regularizer=regularizers.l2(l2_reg))
        self.emb_lr = tf.keras.layers.Embedding(feature_size, 1, embeddings_regularizer=regularizers.l2(l2_reg))

        self.dropout_lr = tf.keras.layers.Dropout(dropout_fm[0])
        self.dropout_fm = tf.keras.layers.Dropout(dropout_fm[1])
        self.dropout_deep = tf.keras.layers.Dropout(dropout_deep[0])
        self.deep = tf.keras.Sequential()
        for i, l in enumerate(deep_layers):
            if use_bn:
                self.deep.add(tf.keras.layers.BatchNormalization())
            self.deep.add(tf.keras.layers.Dense(l, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(l2_reg)))
            if i + 1 < len(dropout_deep) and dropout_deep[i + 1] > 1e-6:
                self.deep.add(tf.keras.layers.Dropout(dropout_deep[i + 1]))
        self.dense_concat = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(l2_reg))

    def zscore_np(self, feature_size, zscore_file=None):
        zscore_mean_np = np.zeros([feature_size, 1], dtype=np.float)
        zscore_var_np = np.ones([feature_size, 1], dtype=np.float)
        if zscore_file != None:
            for line in open(zscore_file).readlines():
                i, m, v = line.strip().split(":")
                zscore_mean_np[int(i)][0] = float(m)
                zscore_var_np[int(i)][0] = float(v)
        return zscore_mean_np, zscore_var_np

    def transform(self,feat):
        if self.is_save_model:
            feat_index = tf.cast(feat[:,0::2], tf.dtypes.int64)
            feat_value = feat[:,1::2]
        else:
            feat_index, feat_value = feat
        return feat_index,feat_value

    def call(self, feat):
        feat_index, feat_value = self.transform(feat)
        #feat_value = tf.where(tf.equal(feat_value, -999.0), 0.0, feat_value)
        transform_feat_index=feat_index
        feat_value=tf.where(tf.equal(feat_value,-999.0),tf.zeros_like(feat_value),feat_value)
        preemb_input = None
        if len(self.preemb_fea) > 0:
            preemb_bool = tf.fill(tf.shape(feat_index), False)
            # preemb_mask = tf.where(feat_index > 0, True, False)
            emb_len = 0
            for x in self.preemb_fea:
                preemb_bool = preemb_bool | (
                            tf.where(feat_index >= x[0], True, False) & tf.where(feat_index < x[1], True, False))
                # preemb_mask = preemb_mask & (tf.where(feat_index < x[0], True, False) | tf.where(feat_index >= x[1], True, False))
                emb_len += x[1] - x[0]
            preemb_mask = tf.where(preemb_bool, False, True)
            # preemb_bool = tf.where(preemb_mask, False , True)
            # print("bo",preemb_bool.numpy()[0])
            # print("b1",preemb_bool.numpy()[1])
            preemb_input = tf.reshape(tf.boolean_mask(feat_value, preemb_bool), shape=[-1, emb_len])
            # print("cc",preemb_input)
            feat_index = tf.where(preemb_mask, feat_index, 0)
            feat_value = tf.where(preemb_mask, feat_value, 0)
        # after_preemb_feat_index=feat_index
        mean = self.zscore_mean(feat_index)
        var = self.zscore_var(feat_index)
        feat_value = tf.math.divide(tf.math.subtract(tf.expand_dims(feat_value, 2), mean), var)
        # limit feat value
        feat_value = tf.clip_by_value(feat_value, -5.0, 5.0)
        # lr part
        lr = tf.reduce_sum(tf.math.multiply(self.emb_lr(feat_index), feat_value), 1)
        lr = self.dropout_lr(lr, training=self.training)
        # fm part
        full_embedding = self.emb_fm(feat_index)
        fm_embedding = tf.math.multiply(full_embedding, feat_value)
        square_sum_fm_embedding = tf.math.square(tf.reduce_sum(fm_embedding, 1))
        sum_square_fm_embedding = tf.reduce_sum(tf.math.square(fm_embedding), 1)
        fm = 0.5 * tf.math.subtract(square_sum_fm_embedding, sum_square_fm_embedding)
        fm = self.dropout_fm(fm, training=self.training)
        # deep part
        single_mask = tf.where(feat_index > 0, True, False)
        # before_multihot_single_mask=single_mask
        def single_mask_while_loop(single_mask, len_multihot_fea, multi_hot_fea_tf):
            def cond(i, single_mask):
                return i < len_multihot_fea
            def body(i, single_mask):
                single_mask = single_mask & (tf.where(feat_index < multi_hot_fea_tf[i, 0], True, False) | tf.where(feat_index >= multi_hot_fea_tf[i, 1], True, False))
                return i + 1, single_mask
            i = tf.constant(0, dtype=tf.int64)
            i, single_mask = tf.while_loop(cond, body, [i, single_mask])
            return single_mask
        single_mask = single_mask_while_loop(single_mask, self.len_multihot_fea, self.multi_hot_fea_tf)
        # for fea in self.multihot_fea:
        #     single_mask = single_mask & (tf.where(feat_index<fea[0], True, False) | tf.where(feat_index>=fea[1], True, False))
        single_feat_index = tf.reshape(tf.boolean_mask(feat_index, single_mask), shape=[1024, self.field_size_single])
        deep = self.emb_fm(single_feat_index)
        def cond(i, deep):
            return i < self.len_multihot_fea
        def body(i, deep):
            mask = tf.where(feat_index >= self.multi_hot_fea_tf[i, 0], 1, 0) * tf.where(feat_index < self.multi_hot_fea_tf[i, 1], 1, 0)
            value = feat_value * tf.cast(tf.expand_dims(mask, 2), tf.dtypes.float32)
            deep = tf.concat([deep, tf.reduce_sum(tf.math.multiply(full_embedding, value), 1, keepdims=True)], axis=1)
            return i + 1, deep
        i = tf.constant(0, dtype=tf.int64)
        i, deep = tf.while_loop(cond, body, loop_vars=[i, deep], shape_invariants=[i.get_shape(), tf.TensorShape([1024, None, 4])])
        # for fea in self.multihot_fea:
        #     mask = tf.where(feat_index >= fea[0], 1, 0) * tf.where(feat_index < fea[1], 1, 0)
        #     value = feat_value * tf.cast(tf.expand_dims(mask, 2), tf.dtypes.float32)
        #     # print(f'in multi-hot-fea, get deep: {deep},\n value:{value},\n full_embediding:{full_embedding}\n reduce_sum: {tf.reduce_sum(tf.math.multiply(full_embedding, value), 1, keepdims=True)}')
        #     deep = tf.concat([deep, tf.reduce_sum(tf.math.multiply(full_embedding, value), 1, keepdims=True)], axis=1)
        deep = tf.reshape(deep, shape=[-1, (self.field_size_single + len(self.multihot_fea)) * self.embedding_size])
        if preemb_input is not None:
            deep = tf.concat([deep, preemb_input], 1)
        # print(f'deep before deep layer: {deep}') # Tensor("deep_fm/Reshape_1:0", shape=(batch, 2984), dtype=float32)
        deep = self.dropout_deep(deep, training=self.training)
        deep = self.deep(deep, training=self.training)
        # print(f'deep after deep layer: {deep}') # Tensor("deep_fm/Reshape_1:0", shape=(batch, out_dim), dtype=float32)
        # print(f'lr: {lr}') # Tensor("deep_fm/dropout/Identity:0", shape=(batch, 1), dtype=float32)
        # print(f'deep: {deep}')
        # print(f'fm: {fm}') # Tensor("deep_fm/dropout_1/dropout/SelectV2:0", shape=(batch, embedding_dim), dtype=float32)
        # concat lr fm & deep
        concat = tf.concat([lr, fm, deep], axis=1)
        out = self.dense_concat(concat)
        return out
