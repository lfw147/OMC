import os
import copy
import pandas as pd
import torch.utils.data as data
import torchvision.models
from torchvision import transforms
from dataset import RafDataset
from model import Model
from utils import *
from resnet import *
import datetime
#from save_test import *

# 获取图片文件路径
# image_folder = r'G:/data/RAF-DB/basic/Image/Image/aligned'
image_folder = r'E:/lfw/data/RAF-DB/basic/Image/Image/aligned'

image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 读取标签文件并创建标签映射
# label_file = r'G:/data/RAF-DB/basic/EmoLabel/list_patition_label.txt'
label_file = r'E:/lfw/data/RAF-DB/basic/EmoLabel/list_patition_label.txt'

label_df = pd.read_csv(label_file, delimiter=' ', header=None, names=['image_name', 'label'])

# 分离训练和测试图片及标签
train_images = [f for f in image_files if 'train' in os.path.basename(f)]
test_images = [f for f in image_files if 'test' in os.path.basename(f)]

train_labels = [int(label_df.loc[label_df['image_name'] == os.path.basename(f).replace('_aligned', ''), 'label'].values[0]) for f in train_images]
test_labels = [int(label_df.loc[label_df['image_name'] == os.path.basename(f).replace('_aligned', ''), 'label'].values[0]) for f in test_images]

train_labels = [label - 1 for label in train_labels]
test_labels = [label - 1 for label in test_labels]
'''
四、优化器，实例化，损失函数，cuda
'''
resnet50_weight='../model/resnet50_ft_weight.pkl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(resnet50_weight).to(device)
loss_c = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def train(train_dl, model, loss_c ,optimizer):
    size = len(train_dl.dataset)
    num_batch = len(train_dl)
    correct, train_loss = 0, 0
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        # output1, output2, _ = model(x)
        # # 计算一致性损失
        # consistency_loss = torch.mean((output2 - output1) ** 2)
        # loss = loss_c(output1, y) +  3*consistency_loss
        output1 = model(x)
        loss=loss_c(output1,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            correct += (output1.argmax(1) == y).type(torch.float).sum().item()
    scheduler.step()
    correct /= size
    train_loss /= num_batch
    return train_loss, correct

def test(test_dl, model, loss_c):
    # hm_list = []
    size = len(test_dl.dataset)
    num_bacth = len(test_dl)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            #output1= model(x)
            output1 = model(x)
            loss = loss_c(output1, y)
            test_loss += loss.item()
            correct += (output1.argmax(1) == y).type(torch.float).sum().item()
            # 将 hm 数据添加到列表中
            # hm_list.append(hm)

        # # 所有的运算都完成之后，将 hm_data 中的所有数据一次性地写入到文件中
        # hm_list = np.array(hm_list)
        # save_data_to_txt("./path/to/your/directory", hm_list)
        scheduler.step()
        test_loss /= num_bacth
        correct /= size
        return test_loss, correct

def pred(test_dl, model, loss_c):
    # hm_list = []
    size = len(test_dl.dataset)
    num_bacth = len(test_dl)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            #output1= model(x)
            output1 = model(x)
            loss = loss_c(output1, y)
            test_loss += loss.item()
            correct += (output1.argmax(1) == y).type(torch.float).sum().item()
            # 将 hm 数据添加到列表中
            # hm_list.append(hm)

        # # 所有的运算都完成之后，将 hm_data 中的所有数据一次性地写入到文件中
        # hm_list = np.array(hm_list)
        # save_data_to_txt("./path/to/your/directory", hm_list)
        scheduler.step()
        test_loss /= num_bacth
        correct /= size
        return test_loss, correct



def fit(epochs, train_dl, test_dl, model, loss_c, optomizer):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    best_acc = 0.0

    best_val_accuracy = 0.0
    patience = 20  # 当验证精确度不再增加时的等待轮数
    counter = 0  # 记录等待的轮数

    # 获取当前时间，并格式化为字符串
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = train(train_dl, model, loss_c, optomizer)
        epoch_test_loss, epoch_test_acc = test(test_dl, model, loss_c)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)

        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        template = ('epoch:{:2d} , train_loss:{:.5f} , train_acc:{:.2f} , test_loss:{:.5f} , test_acc:{:.2f}')
        print(template.format(epoch, epoch_train_loss, epoch_train_acc * 100, epoch_test_loss, epoch_test_acc * 100))
        # 在文件名中添加当前时间
        filename = 'rebuttal_50_{}.txt'.format(current_time)
        with open(filename, 'a') as f:
            f.write(str(epoch)+'_'+str(test_acc)+'\n')
        if epoch_test_acc > best_val_accuracy:
            best_val_accuracy = epoch_test_acc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Validation accuracy did not improve for the last", patience, "epochs. Stopping training.")
                break
    print("Done")
    model.load_state_dict(best_model_wts)
    #model.eval()
    '''
    完整模型的保存和加载
    '''
    path = '../model_test/model_best_{}.pth'.format(current_time)
    print(path)
    torch.save(model, path)

    '''
    恢复模型参数
    '''
    new_model = torch.load(path)
    # new_model.eval()
    epoch_test_loss, epoch_test_acc = pred(test_dl, new_model, loss_c)
    template = ('test_loss:{:.5f} , test_acc:{:.2f}')
    print(template.format(epoch_test_loss, epoch_test_acc * 100))

    return train_loss, train_acc, test_loss, test_acc

epochs=100
def main():
    setup_seed(0)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(scale=(0.02, 0.25))
    ])

    train_dataset = RafDataset(train_images,train_labels ,stage='train', transform=transform)
    test_dataset = RafDataset(test_images, test_labels,stage='test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True)

    fit(epochs, train_loader, test_loader, model, loss_c, optimizer)


if __name__ == '__main__':
    main()