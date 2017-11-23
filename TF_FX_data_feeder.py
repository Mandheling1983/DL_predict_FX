#! /usr/bin/env python3
# coding: utf-8
import csv
import random as random
import numpy as np
import tensorflow as tf

num_of_input_nodes = 6          # 1ステップに入力されるデータ数
num_of_hidden_nodes = 64        # ノード数
num_of_output_nodes = 5         # クラス数
length_of_sequences = 10       # 系列長
# length_of_sequences = 50        # 系列長
num_of_training_epochs = 30000   # 学習ステップ数
# num_of_training_epochs = 50000  # 学習ステップ数
size_of_mini_batch = 300        # バッチサイズ
num_of_prediction_epochs = 100
learning_rate = 0.1
forget_bias = 0.05

def get_batch(batch_size, X, t):
    snum = random.randint(0, len(X) - 1 - batch_size)
    xs = X[snum:snum + batch_size]
    ts = t[snum:snum + batch_size]
    return xs, ts

def make_prediction(nb_of_samples):
    xs, ts = create_data(nb_of_samples, length_of_sequences, True)
    return xs, ts

def create_data(nb_of_samples, sequence_len, Flg):
    """ CSVヘッダ1行読み飛ばし
        2行目以降 2-7カラムがデータ,8カラムをラベルに分離
    """
    if Flg:
        csv_reader = csv.reader(open("DATA_SOURCE/USDJPY_Candlestick_1_h_2017.csv", "r"))
    else:
        csv_reader = csv.reader(open("DATA_SOURCE/USDJPY_Candlestick_1_h_Train.csv", "r"))
    dt = [v for v in csv_reader]
    dt = np.delete(dt, 0, 0)
    dt = np.delete(dt, 0, 1)
    dat = [[float(elm) for elm in v] for v in dt]
    data_body = [[[float(0) for n in range(num_of_input_nodes)] for m in range(sequence_len)] for o in range(len(dat))]
    label_body = [int(0) for mm in range(len(dat))]
    for i in range(len(dat)):
        if i >= sequence_len - 1:
            for k in range(sequence_len):
                for j in range(len(dat[i])):
                    if j <= num_of_input_nodes-1:
                        data_body[i-(sequence_len-1)][sequence_len-1-k][j] = dat[i-k][j]
            label_body[i-(sequence_len-1)] = int(dat[i][j])
    return data_body, label_body

def print_result(p, q):
    print("output: %f, correct: %d" % (p, q))

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

X, t = create_data(size_of_mini_batch, length_of_sequences, False)

with tf.Graph().as_default():
    input_ph = tf.placeholder(tf.float32, \
        [None, length_of_sequences, num_of_input_nodes], name="input")
    supervisor_ph = tf.placeholder(tf.int32, [None], name="supervisor")
    t_on_hot = tf.one_hot(supervisor_ph, depth=num_of_output_nodes, dtype=tf.float32)  # 1-of-Kベクトル
    istate_ph = tf.placeholder(tf.float32, [None, num_of_hidden_nodes * 2], name="istate")
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=False)  # 中間層のセル
    # RNNに入力およびセル設定する
    outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input_ph, dtype=tf.float32, time_major=False)
    # [ミニバッチサイズ,系列長,出力数]→[系列長,ミニバッチサイズ,出力数]
    outputs = tf.transpose(outputs, perm=[1, 0, 2])
    weight = tf.Variable(tf.random_normal([num_of_hidden_nodes, num_of_output_nodes], stddev=learning_rate), name="weight")
    bias = tf.Variable(tf.zeros([num_of_output_nodes]), name="bias")
    output_op = tf.matmul(outputs[-1], weight) + bias  # 出力層
    # Add summary ops to collect data
    w_hist = tf.summary.histogram("weights", weight)
    b_hist = tf.summary.histogram("biases", bias)
    output_hist = tf.summary.histogram("output", output_op)

    pred = tf.nn.softmax(output_op)  # ソフトマックス
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t_on_hot, logits=output_op)
    loss_op = tf.reduce_mean(cross_entropy)  # 誤差関数
    train_step = tf.train.AdamOptimizer().minimize(loss_op)  # 学習アルゴリズム

    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(t_on_hot,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 精度
    y = tf.cast(tf.argmax(pred, 1), tf.int32)
    tf.summary.scalar("loss", loss_op)
    tf.summary.scalar("accuracy", accuracy)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('para')
        if ckpt: # checkpointがある場合
            last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
            print ("load " + last_model)
            saver.restore(sess, last_model) # 変数データの読み込み

        else: # 保存データがない場合
            init = tf.initialize_all_variables()
            sess.run(init) #変数を初期化して実行

        summary_writer = tf.summary.FileWriter("tensorflow_log", graph=sess.graph)

        for epoch in range(num_of_training_epochs):
            inputs, supervisors = get_batch(size_of_mini_batch, X, t)
            train_dict = {
                input_ph:      inputs,
                supervisor_ph: supervisors,
                istate_ph:     np.ones((size_of_mini_batch, num_of_hidden_nodes * 2)),
            }
            sess.run(train_step, feed_dict=train_dict)
            if (epoch+1) % 100 == 0:
                summary_str, train_loss, accuracy_train = sess.run([summary_op, loss_op, accuracy], feed_dict=train_dict)
                print("[TRAIN#%d] loss : %f, accuracy : %f" %(epoch+1, train_loss, accuracy_train))
                summary_writer.add_summary(summary_str, epoch)
            if (epoch+1) % 500 == 0:
                inputs_t, supervisors_t = make_prediction(num_of_prediction_epochs)
                pred_dict_t = {
                    input_ph:  inputs_t,
                    supervisor_ph: supervisors_t,
                    istate_ph: np.ones((len(inputs_t), num_of_hidden_nodes * 2)),
                }
                accuracy_test, test_label = sess.run([accuracy, y], feed_dict=pred_dict_t)
                print("[TEST#%d] accuracy : %f" %(epoch+1, accuracy_test))
        print("[ResultSum] sumlabel : %d" %(sum(test_label)))
        csvfile_w = 'DATA_SOURCE/TEST_Result.csv'
        FW = open(csvfile_w, "w")
        writer = csv.writer(FW, lineterminator='\n')
        writer.writerow(test_label)

        saver.save(sess, "para/fxmodel_ckpt")
