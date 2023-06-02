import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Autoencoder:

    def __init__(self, num_input, num_hidden, num_bottle, wt_stdev=0.1):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_bottle = num_bottle
        self.h1_wts = tf.Variable(tf.random.normal((self.num_input, self.num_hidden), stddev = wt_stdev))
        self.h1_b = tf.Variable(tf.random.normal((self.num_hidden,), stddev = wt_stdev))
        self.b_wts = tf.Variable(tf.random.normal((self.num_hidden, self.num_bottle), stddev = wt_stdev))
        self.b_b = tf.Variable(tf.random.normal((self.num_bottle,), stddev = wt_stdev))
        self.h2_wts = tf.Variable(tf.random.normal((self.num_bottle, self.num_hidden), stddev = wt_stdev))
        self.h2_b = tf.Variable(tf.random.normal((self.num_hidden,), stddev = wt_stdev))
        self.out_wts = tf.Variable(tf.random.normal((self.num_hidden, self.num_input), stddev = wt_stdev))
        self.out_b = tf.Variable(tf.random.normal((self.num_input,), stddev = wt_stdev))
    
    def forward(self, x):
        h1_net_in = x@self.h1_wts + self.h1_b
        h1_net_act = h1_net_in
        # h1_net_act = tf.math.tanh(h1_net_in)
        # h1_net_act = tf.math.sigmoid(h1_net_in)
        # h1_net_act = tf.nn.softmax(h1_net_in)
        # h1_net_act = tf.nn.relu(h1_net_in)

        b_net_in = h1_net_act@self.b_wts + self.b_b
        b_net_act = tf.nn.relu(b_net_in)

        h2_net_in = b_net_act@self.h2_wts + self.h2_b
        h2_net_act = h2_net_in
        # h2_net_act = tf.math.tanh(h2_net_in)
        # h2_net_act = tf.math.sigmoid(h2_net_in)
        # h2_net_act = tf.nn.softmax(h2_net_in)
        # h2_net_act = tf.nn.relu(h2_net_in)

        out_net_in = h2_net_act@self.out_wts + self.out_b
        out_net_act = out_net_in
        # out_net_act = tf.math.tanh(out_net_in)
        # out_net_act = tf.math.sigmoid(out_net_in)
        # out_net_act = tf.nn.softmax(out_net_in)

        return out_net_act

    def get_bottle_neurons(self, x):
        h1_net_in = x@self.h1_wts + self.h1_b
        b_net_in = h1_net_in@self.b_wts + self.b_b

        return b_net_in

    def only_decode(self, bottle_neurons):
        h2_net_in = bottle_neurons@self.h2_wts + self.h2_b
        out_net_in = h2_net_in@self.out_wts + self.out_b

        return out_net_in

    def mse_loss(self, z_net_act, image):
        loss = tf.reduce_mean((z_net_act - image)**2)
        return loss
    
    def early_stopping(self, recent_val_losses, curr_val_loss, patience):
        if len(recent_val_losses) < patience:
            recent_val_losses.append(curr_val_loss)
        else:
            oldest = recent_val_losses.pop(0)
            recent_val_losses.append(curr_val_loss)
            if recent_val_losses[0] == min(recent_val_losses):
                return recent_val_losses, True

        return recent_val_losses, False

    def fit(self, x, x_val, max_epochs, lr, mini_batch_sz, patience, val_every, print_every):
        N, M = x.shape
        adam = tf.optimizers.Adam(lr)
        train_loss_hist = []
        val_loss_hist = []
        recent_val_loss_hist = []
        n_epochs = 0
        stop = False
        while stop == False and n_epochs < max_epochs:
            mini_batch_losses = []
            for b in range(N//mini_batch_sz):
                train_inds = tf.random.uniform([mini_batch_sz, ], 0, N, dtype = tf.int32)
                data_batch = x[train_inds]
                with tf.GradientTape() as tape:
                    netAct = self.forward(data_batch)
                    loss = self.mse_loss(netAct, data_batch)
                grads = tape.gradient(loss, [self.h1_wts, self.h1_b, self.b_wts, self.b_b, self.h2_wts, self.h2_b, self.out_wts, self.out_b])
                adam.apply_gradients(zip(grads, [self.h1_wts, self.h1_b, self.b_wts, self.b_b, self.h2_wts, self.h2_b, self.out_wts, self.out_b]))
                mini_batch_losses.append(loss)
            train_loss_hist.append(tf.reduce_mean(mini_batch_losses).numpy())
            
            if(n_epochs % val_every == 0) or (n_epochs == max_epochs):
                val_netAct = self.forward(x_val)
                val_loss = self.mse_loss(val_netAct, x_val)
                val_loss_hist.append(val_loss.numpy())
                recent_val_loss_hist, stop = self.early_stopping(recent_val_loss_hist, val_loss, patience)

            if(n_epochs % print_every == 0) or (n_epochs == max_epochs):
                print(f'Epoch: {n_epochs} train loss: {train_loss_hist[-1]:.3f} val loss: {val_loss_hist[-1]:.3f}')
            n_epochs += 1

        print(f'Done Training!')

        return train_loss_hist, val_loss_hist, n_epochs