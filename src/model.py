import torch
from torch import nn
from torch.nn import functional as F


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()

        self.conv_1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)

        # reducing outputs from 1152 to 64
        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)

        # Now we have 75 timesteps each with 64 values
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        self.output = nn.Linear(64, 20)

    def forward(self, images, targets=None):
        b_size, c, h, w = images.size()
        # print(b_size, c, h, w)

        x = F.relu(self.conv_1(images))
        # print(x.size())
        x = self.pool_1(x)
        # print(x.size())
        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = self.pool_2(x)  # 1, 64, 18, 75 -> BxFxHxW
        # print(x.size())
        x = x.permute(0, 3, 1, 2)  # 1, 75, 64, 18
        # print(x.size())
        x = x.view(b_size, x.size(1), -1)
        # print(x.size())
        x = self.linear_1(x)
        # print(x.size())
        x = self.drop_1(x)
        x, _ = self.gru(x)
        # print(x.size())
        x = self.output(x)
        # print(x.size())

        # time x b_size x value -> expected for CTC loss
        x = x.permute(1, 0, 2)
        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(b_size,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(b_size,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None


if __name__ == '__main__':
    cm = CaptchaModel(19)
    img = torch.rand(5, 3, 75, 300)
    target = torch.randint(1, 20, (5, 5))
    x, loss = cm(img, target)
