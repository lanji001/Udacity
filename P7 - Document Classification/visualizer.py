import matplotlib.pyplot as plt

# 根据训练过程中的记录数据，绘制准确率的提升过程
# @param: history，训练过程中的记录数据
def plot_accuracy(history):
    plt.plot(history.history['acc'], color="r")
    plt.plot(history.history['val_acc'], color="g")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

# 根据训练过程中的记录数据，绘制损失值的下降过程
# @param: history，训练过程中的记录数据
def plot_loss(history):
    plt.plot(history.history['loss'], color="r")
    plt.plot(history.history['val_loss'], color="g")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()
