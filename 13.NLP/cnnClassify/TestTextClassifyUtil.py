from TextClassifyUtil import *
import sys

is_py3 = True
if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

categories = ["Sports", "Music", "Entertainment", "Star"]
_, categoryIds = convertLabelsToLabelId(categories)
print("categoryIds ", categoryIds)

base_dir = './../../../data/nlp/dataCnnNlp/cnews'
train_dir = os.path.join(base_dir, 'cnews.val.txt')  # cnews.train.txt
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

categories, cat_to_id = read_category()          # 制作分类目录
words, word_to_id = read_vocab(vocab_dir)

params = CNNTextTFParams()    

cnnTextTF = CNNTextTF(params)

cnnTextTF.buildGraph()

x_train, y_train = process_file_get_data(train_dir, word_to_id, cat_to_id, params.seq_length)
cnnTextTF.train(x_train, y_train)

x_batch, y_batch = next(batch_iterator(x_train, y_train, params.batch_size))
print("x_batch ", x_batch)
print("y_batch ", y_batch)

rnnparams = RNNTextTFParams()    
rnnTextTF = RNNTextTF(rnnparams)

rnnTextTF.buildGraph()
rnnTextTF.train(x_train, y_train)






