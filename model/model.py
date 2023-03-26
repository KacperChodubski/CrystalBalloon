import torch
import torch.nn as nn

class PredictionModel(nn.Module):
    def __init__(self):
        super(PredictionModel, self).__init__()

        #self.writer = SummaryWriter("runs/logs")

        input_size = 6
        hidden_size = 8

        # lat layers
        self.l1_1 = nn.Linear(input_size, hidden_size)
        self.l1_2 = nn.Linear(hidden_size, hidden_size)
        self.l1_4 = nn.Linear(hidden_size, 1)

        # lon layers
        self.l2_1 = nn.Linear(input_size, hidden_size)
        self.l2_2 = nn.Linear(hidden_size, hidden_size)
        self.l2_4 = nn.Linear(hidden_size, 1)

        # alt layers
        self.l3_1 = nn.Linear(input_size, hidden_size)
        self.l3_2 = nn.Linear(hidden_size, hidden_size)
        self.l3_4 = nn.Linear(hidden_size, 1)


        

    def forward(self, x):

        out1 = self.l1_1(x)
        out1 = self.l1_2(out1)
        out1 = self.l1_4(out1)

        out2 = self.l2_1(x)
        out2 = self.l2_2(out2)
        out2 = self.l2_4(out2)

        out3 = self.l3_1(x)
        out3 = self.l3_2(out3)
        out3 = self.l3_4(out3)

        return {'lat': out1, 'lon': out2, 'alt': out3}
