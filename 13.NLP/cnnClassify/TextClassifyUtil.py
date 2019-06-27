import sys
from collections import Counter

import numpy as np
import tensorflow as tf

import os
import sys
import time

import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]       # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    word_to_id = dict(zip(words, range(len(words))))                      # 将word做成字典，key是每一行，value是行数
    return words, word_to_id

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels

# ["Sports", "Music", "Entertainment", "Star"]
# => {"Sports": 0, "Music":1, "Entertainment": 2, "Star": 3}
def convertLabelsToLabelId(labels):
    
    labelsId = dict(zip(labels, range(len(labels))))

    return labels, labelsId

def process_file_get_data(train_dir, word_to_id, cat_to_id,seq_length):

    contents, labels = read_file(train_dir)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    print("data_id 1 ", data_id[1])
    print("label_id 1 ", label_id[1])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, seq_length)   # 将句子都变成600大小的句子，超过600的从后边开始数，去除前边的
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    print("x_pad 1  ", x_pad[1])
    print("y_pad 1 ", y_pad[1])

    return x_pad, y_pad

def batch_iterator(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

class CNNTextTFParams(object):
    embedding_dim = 64   # 词向量维度
    seq_length =  600  # 序列长度
    num_classes = 10  # 类别数

    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸

    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard



class CNNTextTF(object):

    def __init__(self, params):
        self.params = params
        return None

    def buildGraph(self):
        #initialize input data
        self.input_x = tf.placeholder(tf.int32, [None, self.params.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.params.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        #word vector embedding input
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.params.vocab_size, self.params.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # CNN layer
        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_inputs, self.params.num_filters, self.params.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # dense layer，=> dropout => relu 
            fc = tf.layers.dense(gmp, self.params.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # classifier 
            self.logits = tf.layers.dense(fc, self.params.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # predict the class

        with tf.name_scope("optimize"):
            # cross entropy loss function 
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # optimizer 
            self.optim = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            #accuracy
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    

        return None

    def train(self, x_train, y_train):
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

        flag = False
        for epoch in range(self.params.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iterator(x_train, y_train, self.params.batch_size)
            
            for x_batch, y_batch in batch_train:
                feed_dict = {
                    self.input_x: x_batch,
                    self.input_y: y_batch,
                    self.keep_prob: self.params.dropout_keep_prob
                }
                
                if total_batch % self.params.print_per_batch == 0:
                    #print("x_batch ", x_batch)
                    #print("y_batch ", y_batch)
                    loss_train, acc_train = session.run([self.loss, self.acc], feed_dict=feed_dict)
                    print("CNNTextTF total_batch ", total_batch)
                    print("CNNTextTF loss_train ", loss_train)

                session.run(self.optim, feed_dict=feed_dict)  # 运行优化 真正开始运行,因为是相互依赖，倒着找的
                total_batch += 1        
        print("CNNTextTF final loss_train ", loss_train)
        return None

class RNNTextTFParams(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class RNNTextTF(object):
    """文本分类，RNN模型"""
    def __init__(self, params):
        self.params = params

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.params.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.params.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def buildGraph(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.params.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.params.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.params.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embeddingrnn = tf.get_variable('embeddingrnn', [self.params.vocab_size, self.params.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embeddingrnn, self.input_x)
            

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.params.num_layers)]         # 定义cell
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True) # 将两层的lstm组装起来

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)  # _outputs表示最后一层的输出【？，600,128】；"_"：表示每一层的最后一个step的输出，也就是2个【？，128】，几层就有几个【？，128】
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("scorernn"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.params.hidden_dim, name='fc1rnn')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.params.num_classes, name='fc2rnn')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimizernn"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracyrnn"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return None
    
    def train(self, x_train, y_train):
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

        flag = False
        for epoch in range(self.params.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iterator(x_train, y_train, self.params.batch_size)
            
            for x_batch, y_batch in batch_train:
                feed_dict = {
                    self.input_x: x_batch,
                    self.input_y: y_batch,
                    self.keep_prob: self.params.dropout_keep_prob
                }
                
                if total_batch % self.params.print_per_batch == 0:
                    loss_train, acc_train, y_predict = session.run([self.loss, self.acc, self.y_pred_cls], feed_dict=feed_dict)
                    print('x_batch shape:',x_batch.shape)
                    print('y_batch shape:',y_batch.shape)
                    print("RNNTextTF total_batch ", total_batch)
                    print("RNNTextTF loss_train ", loss_train)
                    print('y_predict shape:',y_predict.shape)
                    print('y_predict:',y_predict)


                session.run(self.optim, feed_dict=feed_dict)  # 运行优化 真正开始运行,因为是相互依赖，倒着找的
                total_batch += 1        
        print("Text RNN final loss_train ", loss_train)        



        return None