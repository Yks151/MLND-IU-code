import torch
from torch.utils.data import DataLoader
from data import LungNoduleDataset
from MLND_model import MLND_IU
from losses import MultiStageLoss
import argparse

def train(args):
    # 数据加载
    train_set = LungNoduleDataset(args.train_paths, mode='train')
    val_set = LungNoduleDataset(args.val_paths, mode='val')

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    # 模型初始化
    model = MLND_IU().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = MultiStageLoss()

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            images = batch['image'].cuda()  # (B,5,H,W)
            labels = batch['label'].cuda()  # (B,H,W)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].cuda()
                labels = batch['label'].cuda()
                outputs, _ = model(images)
                val_loss += criterion(outputs, labels).item()

        print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        scheduler.step()

        # 保存模型
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch{epoch+1}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_paths', type=str, required=True)
    parser.add_argument('--val_paths', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=175)
    args = parser.parse_args()

    train(args)