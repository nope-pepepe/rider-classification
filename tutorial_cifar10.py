# -*- coding: utf-8 -*-

"""
CIFAR10のチュートリアルコード
今回はGPUを使用しない
"""

#ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
CNNの定義
畳み込み2回→全結合層3回のネットワーク
最初に__init__部分でネットワークの形を定義し、ネットワークに画像(x)を入力することで出力が返ってくる
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#画像表示関数の定義
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(model, trainloader, optimizer, device, criterion, epoch, args):
    model.train()   #model訓練モードへ移行
    correct = 0
    total = 0
    running_loss = 0.0  #epoch毎の誤差合計

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        #出力計算
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        #Accuracy計算用素材
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        #勾配計算とAdamStep
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    accuracy = correct / total * 100.0
    print("Train Epoch:{:>3} Acc:{:3.1f}% Loss:{:.4f}".format(epoch, accuracy, running_loss))

def validation(model, valloader, device, criterion, args):
    model.eval()    #モデル推論モードへ移行
    correct = 0
    total = 0
    running_loss = 0.0  #epoch毎の誤差合計

    with torch.no_grad():   #勾配計算を行わない状態
        for i, (inputs, labels) in enumerate(valloader):
            inputs, labels = inputs.to(device), labels.to(device)
            #出力計算
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #Accuracy計算用素材
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            running_loss += loss.item()

        accuracy = correct / total * 100.0
        print("Val Acc:{:3.1f}% Loss:{:.4f}".format(accuracy, running_loss))

    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0, help="使用GPU番号 空き状況はnvidia-smiで調べる")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="データセット周回数")
    parser.add_argument("-b", "--batchsize", type=int, default=16, help="ミニバッチサイズ")
    parser.add_argument("-l", "--learningrate", type=float, default=0.001, help="学習率")
    parser.add_argument("--num-worker", type=int, default=4, help="CPU同時稼働数 あまり気にしなくてよい")
    parser.add_argument("--modelname", type=str, default="bestmodel.pth", help="保存モデル名")
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu))   #GPUの設定

    #画像を正規化する関数の定義
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    """
    データセットの読み込み 初回はデータセットをダウンロードするためネットにつながっている必要あり
    train=Trueなら学習用画像5万枚読み込み
    train=Falseならテスト用画像1万枚読み込み
    batchsizeは学習時に一度に扱う枚数
    shuffleは画像をランダムに並び替えるかの設定 test時はオフ
    num_workersはCPU使用数みたいなもの　気にしないで良い
    """
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,
                                            shuffle=True, num_workers=args.num_worker)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize,
                                            shuffle=False, num_workers=args.num_worker)

    #画像のクラス 全10クラス
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #ランダムな画像の選択(表示するため)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    #正解ラベルの表示
    print(' '.join('%5s' % classes[labels[j]] for j in range(args.batchsize)))
    #画像の表示
    imshow(torchvision.utils.make_grid(images))
    
    #ネットワークの定義
    #net(input)とすることで画像をネットワークに入力できる
    net = Net()
    net = net.to(device)    #modelをGPUに送る

    """
    ここから誤差関数の定義
    """

    #SoftmaxCrossEntropyLossを使って誤差計算を行う。計算式はググってください。
    criterion = nn.CrossEntropyLoss()
    #学習器の設定 lr:学習率
    #SGDやAdamという学習器がよく使われる
    optimizer = optim.SGD(net.parameters(), lr=args.learningrate, momentum=0.9)

    """
    ここから訓練ループ
    epoch       :同じデータに対し繰り返し学習を行う回数。
    """
    max_acc = 0.0

    for epoch in range(args.epoch):
        running_loss = 0.0  #epoch毎の誤差の合計。
        train(net, trainloader, optimizer, device, criterion, epoch, args)
        val_accuracy = validation(net, valloader, device, criterion, args)

        #validationの成績が良ければモデルを保存
        if val_accuracy > max_acc:
            torch.save(net.state_dict(), args.modelname)

    print('Finished Training')

    """
    ここからは学習したモデルで出力の確認
    """
    dataiter = iter(valloader)
    images, labels = dataiter.next()

    # print images
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(args.batchsize)))
    
    
    net.load_state_dict(torch.load(args.modelname))
    
    inputs = images.to(device)
    outputs = net(inputs)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(args.batchsize)))
    imshow(torchvision.utils.make_grid(images))


    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(args.batchsize):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10): #各クラスのAccuracy表示
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == "__main__":
    main()