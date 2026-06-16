import tensorflow as tf
import numpy as np
import math
import pdb

""" ======== Non Layers ========= """

def initializer(init, shape):
    if init == "zero":
        return tf.zeros(shape, dtype=tf.float32)
    elif init == "he":
        fan_in = np.prod(shape[0:-1])
        std = 1 / np.sqrt(fan_in)
        return tf.random.uniform(shape, minval=-std, maxval=std, dtype=tf.float32)




class GCN_NODE_WEIGHT(tf.keras.Model):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.Wc = tf.Variable(initializer("he", (in_dims, out_dims)), trainable=True) # (in_dims, out_dims)
        self.Wn = tf.Variable(initializer("he", (in_dims, out_dims)), trainable=True) # (in_dims, out_dims)
        self.We = tf.Variable(initializer("he", (2, out_dims)), trainable=True) # (2, out_dims)
        self.q = tf.Variable(initializer("he", (out_dims, 1)), trainable=True)
        self.b = tf.Variable(initializer("zero", (out_dims,)), trainable=True)


    def call(self, x, adj, edge, training):
        Zc = tf.matmul(x, self.Wc)
        v_Wn = tf.matmul(x, self.Wn)
        
        e_We = tf.tensordot(edge, self.We, axes=[[2], [0]]) # (n_verts, n_nbors, filters)

        
        nh_sizes = tf.expand_dims(tf.math.count_nonzero(adj + 1, axis=1, dtype=tf.float32), -1)
        neighbor = tf.gather(v_Wn, adj) + e_We
        # neighbor = tf.gather(v_Wn, adj)
        # weight = tf.nn.softmax(tf.matmul(neighbor, self.q), axis=1)
        weight = tf.nn.softmax(tf.matmul(neighbor, self.q))

        Zn = tf.divide(tf.reduce_sum(neighbor * weight, 1), tf.maximum(nh_sizes, tf.ones_like(nh_sizes))) # (in_dims, out_dims)
        # Zn = tf.reduce_sum(neighbor * weight, 1)
        
        h = tf.nn.relu(Zn + Zc+ self.b)
        if training:
            h = tf.nn.dropout(h, 0.6)

        return h


class Dense(tf.keras.Model):
    def __init__(self, in_dims, out_dims, is_relu):
        super().__init__()
        self.W = tf.Variable(initializer("he", [in_dims, out_dims]), trainable=True)
        self.b = tf.Variable(initializer("zero", [out_dims]), trainable=True)
        self.is_relu = is_relu

    def call(self, x, training):
        if training:
            x = tf.nn.dropout(x, 0.6)
        Z = tf.matmul(x, self.W) + self.b
        if self.is_relu:
            Z = tf.nn.relu(Z)

        return Z


class Bilinear(tf.keras.Model):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super(Bilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.weight = tf.Variable(initializer("he", [out_features, in1_features, in2_features]), trainable=True)

        if bias:
            self.bias = tf.Variable(initializer("zero", [out_features]), trainable=True)
        else:
            self.bias = None

    def call(self, x1, x2):
        node = x1.shape[0]
        
        w = tf.expand_dims(self.weight, 0)
        w = tf.tile(w, tf.stack([node, 1, 1, 1]))

        x1 = tf.expand_dims(x1, 1)
        x1 = tf.expand_dims(x1, 3)
        x1 = tf.tile(x1, tf.stack([1, self.out_features, 1, self.in2_features]))

        x2 = tf.expand_dims(x2, 1)
        x2 = tf.tile(x2, tf.stack([1, self.out_features, 1]))

        buff = tf.reduce_sum(tf.multiply(x1, w), 2)
        out = tf.reduce_sum(tf.multiply(buff, x2), 2)
        if self.bias is not None:
            out += self.bias
        return out


class Affinity(tf.keras.Model):
    def __init__(self, hid_dim=512):
        super(Affinity, self).__init__()
        self.hid_dim = hid_dim
        self.A = tf.Variable(initializer("he", [hid_dim, hid_dim]), trainable=True)
    
    def call(self, x1, x2):
        M = tf.matmul(x1, (self.A + tf.transpose(self.A, [1, 0])) / 2)
        # M = tf.matmul(x1, self.A)
        M = tf.matmul(M, tf.transpose(x2, [1, 0]))
        return M


class Discriminator(tf.keras.Model):
    def __init__(self, out_dim):
        super(Discriminator, self).__init__()
        self.bilinear = Bilinear(out_dim, out_dim, 1)

    def call(self, x1, x2):
        c = tf.reduce_mean(x1, axis=0, keepdims=True)
        c = tf.nn.relu(c)
        c = tf.tile(c, (x1.shape[0], 1))

        # pdb.set_trace()
        # sc_1 = tf.reduce_sum(self.affinity(x1, c), axis=-1)
        # sc_2 = tf.reduce_sum(self.affinity(x2, c), axis=-1)
        sc_1 = tf.squeeze(self.bilinear(x1, c))
        sc_2 = tf.squeeze(self.bilinear(x2, c))

        logits = tf.concat([sc_1, sc_2], axis=0)
        # logits = tf.nn.sigmoid(logits)

        return logits


class PW_classifier(tf.keras.Model):
    def __init__(self, in_dims, gcn_layer_num, gcn_config):
        super().__init__()
        self.gcn_layer_num = gcn_layer_num
        
        self.gcn1 = (GCN_NODE_WEIGHT(in_dims, 512), GCN_NODE_WEIGHT(in_dims, 512))
        self.gcn2 = (GCN_NODE_WEIGHT(512, 512), GCN_NODE_WEIGHT(512, 512))
        # self.gcn3 = (GCN_NODE_WEIGHT(512, 512), GCN_NODE_WEIGHT(512, 512))
        # self.gcn4 = (GCN_NODE_WEIGHT(512, 512), GCN_NODE_WEIGHT(512, 512))


        self.discriminator = Discriminator(512)
        # self.enc2 = (Encoder(256, 512), Encoder(256, 512))

        self.gcn_cross = Dense(1024, 512, True)
        self.gcn_final = (GCN_NODE_WEIGHT(512, 512), GCN_NODE_WEIGHT(512, 512))
        # self.enc1_final = (Encoder([512, 512], 1), Encoder([512, 512], 1))

        self.dense1 = Dense(1024, 512, True)
        self.dense2 = Dense(512, 1, False)

        self.affinity = Affinity(512)


    @tf.function
    def call(self, x0, adj0, e0, x1, adj1, e1, examples, training):
        x0 = tf.cast(x0, dtype=tf.float32)
        x1 = tf.cast(x1, dtype=tf.float32)
        e0 = tf.cast(e0, dtype=tf.float32)
        e1 = tf.cast(e1, dtype=tf.float32)
        
        # shuffle
        e0_shuffle = tf.gather(e0, tf.random.shuffle(tf.range(tf.shape(e0)[0])))
        e1_shuffle = tf.gather(e1, tf.random.shuffle(tf.range(tf.shape(e1)[0])))
        adj0_shuffle = tf.gather(adj0, tf.random.shuffle(tf.range(tf.shape(adj0)[0])))
        adj1_shuffle = tf.gather(adj1, tf.random.shuffle(tf.range(tf.shape(adj1)[0])))

        # inter graph
        # first layer
        x00 = self.gcn1[0](x0, adj0, e0, training)
        x00_shuf = self.gcn1[0](x0, adj0_shuffle, e0_shuffle, training)

        x10 = self.gcn1[1](x1, adj1, e1, training)
        x10_shuf = self.gcn1[1](x1, adj1_shuffle, e1_shuffle, training)

        # # second layer
        x01 = self.gcn2[0](x00, adj0, e0, training)
        x01_shuf = self.gcn2[0](x00_shuf, adj0_shuffle, e0_shuffle, training)

        x11 = self.gcn2[1](x10, adj1, e1, training)
        x11_shuf = self.gcn2[1](x10_shuf, adj1_shuffle, e1_shuffle, training)

        # # third layer
        # x02 = self.gcn3[0](x00 + x01, adj0, e0, training)
        # x02_shuf = self.gcn3[0](x00_shuf + x01_shuf, adj0_shuffle, e0_shuffle, training)

        # x12 = self.gcn3[1](x10 + x11, adj1, e1, training)
        # x12_shuf = self.gcn3[1](x10_shuf + x11_shuf, adj1_shuffle, e1_shuffle, training)

        # # forth layer
        # x03 = self.gcn4[0](x00 + x01 + x02, adj0, e0, training)
        # x03_shuf = self.gcn4[0](x00_shuf + x01_shuf + x02_shuf, adj0_shuffle, e0_shuffle, training)

        # x13 = self.gcn4[1](x10 + x11 + x12, adj1, e1, training)
        # x13_shuf = self.gcn4[1](x10_shuf + x11_shuf + x12_shuf, adj1_shuffle, e1_shuffle, training)

        x0 = x00 + x01 # + x02 + x03
        x1 = x10 + x11 # + x12 + x13

        x0_shuf = x00_shuf + x01_shuf # + x02_shuf + x03_shuf
        x1_shuf = x10_shuf + x11_shuf # + x12_shuf + x13_shuf
        # x0 = x01
        # x1 = x11
        # x0 = self.gcn3[0](x0, adj0, e0, training)
        # x1 = self.gcn3[1](x1, adj1, e1, training)
        # x0 = self.gcn4[0](x0, adj0, e0, training)
        # x1 = self.gcn4[1](x1, adj1, e1, training)
        # x0, logits = self.enc2[0](x0, adj0, e0, True, training)
        # x1, logits = self.enc2[1](x1, adj1, e1, True, training)

        logits0_0 = self.discriminator(x0, x0_shuf)
        logits1_0 = self.discriminator(x1, x1_shuf)

        for i in range(2):
            aff = self.affinity(x0, x1)
            x0_encoding = tf.matmul(tf.nn.softmax(aff), x1)
            x1_encoding = tf.matmul(tf.nn.softmax(tf.transpose(aff, [1, 0])), x0)
            # x0_new = self.gcn_cross(tf.concat((x0, x0), axis=-1), training)
            # x1_new = self.gcn_cross(tf.concat((x1, x1), axis=-1), training)
            x0_new = self.gcn_cross(tf.concat((x0, x0_encoding), axis=-1), training)
            x1_new = self.gcn_cross(tf.concat((x1, x1_encoding), axis=-1), training)
            x0 = x0_new
            x1 = x1_new
        
            x0 = self.gcn_final[0](x0, adj0, e0, training)
            x1 = self.gcn_final[1](x1, adj1, e1, training)

            # x0_shuffle = self.gcn_final[0](x0, adj0_shuffle, e0_shuffle, training)
            # x1_shuffle = self.gcn_final[1](x1, adj1_shuffle, e1_shuffle, training)

        # logits0_1 = self.discriminator(x0, x0_shuffle)
        # logits1_1 = self.discriminator(x1, x1_shuffle)
        
        # logits0 =  tf.nn.sigmoid(logits0_0 + logits0_1)
        # logits1 =  tf.nn.sigmoid(logits1_0 + logits1_1)
        logits0 =  tf.nn.sigmoid(logits0_0)
        logits1 =  tf.nn.sigmoid(logits1_0)
        # merge layer
        out1 = tf.gather(x0, examples[:, 0])
        out2 = tf.gather(x1, examples[:, 1])
        output1 = tf.concat([out1, out2], axis=0)
        output2 = tf.concat([out2, out1], axis=0)
        output = tf.concat((output1, output2), axis=1)

        # dense layer
        output = self.dense1(output, training)
        out = self.dense2(output, training)

        # # average layer
        out = tf.reduce_mean(tf.stack(tf.split(out, 2)), 0)
        out = tf.nn.sigmoid(out)

        return (out, (logits0, logits1), aff)


class Weight_Cross_Entropy(tf.keras.Model):
    def __init__(self, pn_ratio=0.5):
        super().__init__()
        self.pn_ratio = pn_ratio
        self.cerition = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, preds, labels, labels2):
        labels = tf.cast(labels, dtype=tf.float32)
        l_lbl, r_lbl = labels2
        preds, (logits0, logits1), _ = preds

        loss_l = self.cerition(logits0, l_lbl)
        loss_r = self.cerition(logits1, r_lbl)

        scale_vector = (self.pn_ratio * (labels - 1) / -2) + ((labels + 1) / 2)
        labels = (labels + 1) / 2
        labels = tf.expand_dims(labels, -1)
        
        loss = tf.keras.losses.binary_crossentropy(labels, preds)
        loss = tf.reduce_mean(loss * scale_vector)

        final_loss = loss  + 5 * (tf.reduce_mean(loss_l) + tf.reduce_mean(loss_r))

        return  final_loss

     
