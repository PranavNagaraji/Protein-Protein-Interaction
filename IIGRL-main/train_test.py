import numpy as np
from configuration import printt
from model import Weight_Cross_Entropy
import tensorflow as tf
import os
import pdb


class TrainTest:
    def __init__(self, results_log, gcn_layer_num, results_processor=None, epochs=200, batch_size=256):
        self.results_processor = results_processor
        self.gcn_layer_num = gcn_layer_num
        self.results_log = results_log
        self.epochs = epochs
        self.batch_size = batch_size

    def fit_model(self, data, model, cerition, optimizer):
        """
        trains model by iterating minibatches for specified number of epochs
        """
        printt("Fitting Model")
        first_experiment = True
        curRes = 0.
        ss = 0.
        # train for specified number of epochs
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(data["train"], model, self.batch_size, cerition, optimizer)
        # calculate train and test metrics
            save_path = os.path.join(os.path.dirname(self.results_log), "model","droup06_2")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if epoch <120:
                ss = ss + 0.01
                model.save_weights(os.path.join(save_path, "{:}.weights".format(ss)))
            if epoch >= 120:
                headers, result = self.results_processor.process_results(data, model, "epoch_" + str(epoch))
                # save model
                
                if result[7] > curRes:
                    curRes = result[7]
                    # model.save 
                    model.save_weights(os.path.join(save_path, "{:}.weights".format(curRes)))
                    # model.save_weights(os.path.join(save_path, "my_checkpoint"))
                    # model.save('keras_model_tf_version', save_format='tf')

                self.results_processor.reset()

                # write headers to file if haven't already
                if first_experiment:
                    with open(self.results_log, 'a') as f:
                        f.write("{}\n".format(",".join(["layer", "epoch"] + headers)))
                    first_experiment = False
                # write results to file
                with open(self.results_log, 'a') as f:
                    f.write("{}, {}, {}\n".format(self.gcn_layer_num, epoch, ",".join([str(r) for r in result])))
        # model.close()
        return headers, result



    def train_epoch(self, data, model, minibatch_size, cerition, optimizer, dropout_keep_prob=0.5):
        """
        Trains model for one pass through training data, one protein at a time
        Each protein is split into minibatches of paired examples.
        Features for the entire protein is passed to model, but only a minibatch of examples are passed
        """
        prot_perm = np.random.permutation(len(data))
        # pdb.set_trace()
        # loop through each protein
        sum_loss = 0.
        n_batch = 0
        for protein in prot_perm:
            # extract just data for this protein
            prot_data = data[protein]
            # pdb.set_trace()
            pair_examples = prot_data["label"]
            n = len(pair_examples)
            shuffle_indices = np.random.permutation(np.arange(n)).astype(int)
            # shuffle_indices = np.concatenate((shuffle_indices, shuffle_indices))
            # loop through each minibatch
            for i in range(int(n / minibatch_size)):
                # extract data for this minibatch
                index = int(i * minibatch_size)
                examples = pair_examples[shuffle_indices[index: index + minibatch_size]]
                batch_examples = examples
                labels = examples[:,2]

                # train the model

                l_lbl_1 = tf.ones(prot_data['l_vertex'].shape[0])
                l_lbl_2 = tf.zeros(prot_data['l_vertex'].shape[0])
                l_lbl = tf.concat((l_lbl_1, l_lbl_2), axis=0)

                r_lbl_1 = tf.ones(prot_data['r_vertex'].shape[0])
                r_lbl_2 = tf.zeros(prot_data['r_vertex'].shape[0])
                r_lbl = tf.concat((r_lbl_1, r_lbl_2), axis=0)

                with tf.GradientTape() as tape:
                    preds = model(prot_data['l_vertex'], prot_data['l_hood_indices'].squeeze(), prot_data['l_edge'], 
                                    prot_data['r_vertex'], prot_data['r_hood_indices'].squeeze(), prot_data['r_edge'], 
                                    batch_examples, True)
                   
                    loss = cerition(preds, labels, (l_lbl, r_lbl))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

                n_batch += 1
                sum_loss += loss.numpy()
        printt("epoch avg loss {:}".format(sum_loss / len(data)))

