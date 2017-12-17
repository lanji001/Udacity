import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
import numpy as np

# 通过sklearn加载20newsgroups数据包
# @param:  categories，新闻话题
# @return: 训练集，测试集，数据集
def load_data(categories=None):
    newsgroups_train = fetch_20newsgroups(subset='train', 
                                          remove=('headers', 'footers', 'quotes'), 
                                          categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', 
                                         remove=('headers', 'footers', 'quotes'), 
                                         categories=categories)
    newsgroups_data = fetch_20newsgroups(subset='all',
                                         remove=('headers', 'footers', 'quotes'), 
                                         categories=categories)
    return newsgroups_train, newsgroups_test, newsgroups_data


# 对训练集和测试集向量化
# @param:  vectorizer，向量器；train_data，训练集；test_data，测试集
# @return: 经过向量化的训练集数据，训练集标签，测试集数据，测试集标签
def fit_transform(vectorizer, train_data, test_data):
    X_train = vectorizer.fit_transform(train_data.data)
    y_train = train_data.target

    X_test = vectorizer.transform(test_data.data)
    y_test = test_data.target

    return X_train, y_train, X_test, y_test


# 评估结果
# @param: X_test，测试集数据；y_test，测试集标签；model，模型
def valuate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print("accuracy score: %.2f" % metrics.accuracy_score(y_test, y_pred))
    print("f1-score: %.2f" % metrics.f1_score(y_test, y_pred, average='macro'))
    print("precision score: %.2f" % metrics.precision_score(y_test, y_pred, average='macro'))
    print("recall score: %.2f" % metrics.recall_score(y_test, y_pred, average='macro'))

    
# 网格搜索
# @param: model，模型；parameters，超参数；X_train，训练数据； y_train，训练集标签，
#         scoring，选择最优模型的评分方式； cv，数据集分成几份； n_jobs，用于训练的核数
def grid_search(model, parameters, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1):
    best_clf = GridSearchCV(model, parameters, scoring='accuracy', cv=5, n_jobs=-1)
    best_clf.fit(X_train, y_train)
    
    # 打印最优模型的超参数
    print(best_clf.best_params_)

    
# 词向量话
# @param:  tokenizer，词向量器；newsgroups_data，数据集；newsgroups_train，训练集；newsgroups_test，测试集 \
#          MAX_SEQUENCE_LENGTH，每篇新闻的最长单词数；EMBEDDING_DIM，词向量的维度
# @return: 经过向量化的训练集数据，训练集标签，测试集数据，测试集标签
def vectorize_data(tokenizer, newsgroups_data, newsgroups_train, newsgroups_test, \
                   MAX_SEQUENCE_LENGTH=800, EMBEDDING_DIM=100):
    train_sequence = tokenizer.texts_to_sequences(newsgroups_train.data)
    X_train = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = to_categorical(np.asarray(newsgroups_train.target))
    print('Shape of train data tensor:', X_train.shape)
    print('Shape of train label tensor:', y_train.shape)

    test_sequence = tokenizer.texts_to_sequences(newsgroups_test.data)
    X_test = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = to_categorical(np.asarray(newsgroups_test.target))
    print('Shape of test data tensor:', X_test.shape)
    print('Shape of test label tensor:', y_test.shape)

    return X_train, y_train, X_test, y_test


# 生成词嵌入层
# @param:  word_index，词向量模型中词的索引；text8_model，基于text8数据包训练出的词嵌入模型 \
#          MAX_SEQUENCE_LENGTH，每篇新闻的最长单词数；EMBEDDING_DIM，词向量的维度
# @return: 词嵌入层
def generate_embedding_layer(word_index, text8_model, MAX_SEQUENCE_LENGTH=800, EMBEDDING_DIM=100):
    
    # 词矩阵初始化为0
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        # 如果text8_model包含该词，使用词嵌入模型中该词的词向量
        if word in text8_model:
            embedding_matrix[i] = np.asarray(text8_model[word], dtype='float32')

    # 调用keras.layers.Embedding包，生成词嵌入层
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM, 
                                weights = [embedding_matrix],
                                input_length = MAX_SEQUENCE_LENGTH,
                                trainable = False)
    return embedding_layer


# 生成卷积神经网络模型
# @param:  embedding_layer，词嵌入层
# @return: model，卷积神经网络
def get_cnn_model(embedding_layer, y_train, EMBEDDING_DIM=100):
    model = Sequential()
    # 加载词嵌入层
    model.add(embedding_layer)
    model.add(Dropout(0.2))

    # 第一个卷积层＋最大池化层＋丢弃层组合
    model.add(Conv1D(128, 2, padding='same', activation='relu', strides=1))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    # 第二个卷积层＋最大池化层＋丢弃层组合
    model.add(Conv1D(256, 4, padding='same', activation='relu', strides=2))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.5))

    # 全连接层
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(EMBEDDING_DIM, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # 定义模型的训练方式
    model.compile(loss='categorical_crossentropy',
                                   optimizer='rmsprop',
                                   metrics=['acc'])
    return model


if __name__ == "__main__":
    # execute only if run as a script
    main()
