import torch
from torch import nn, optim
from typing import Optional

# Efficient Long Term Memory: ELTM
class ELTMcell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ELTMcell, self).__init__()
        # 重置输入权重矩阵初始化
        self.input_in = nn.Linear(input_size, hidden_size * 2, bias=False)
        self.hidden_in = nn.Linear(hidden_size, hidden_size * 2)
        self.only_x = nn.Linear(input_size, hidden_size, bias=False)
        self.only_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # 更新输出权重矩阵初始化
        self.input_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.candidate_hidden_out = nn.Linear(hidden_size, hidden_size)
        # 分支输出参数初始化
        self.alpha_p = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_p = nn.Linear(hidden_size, hidden_size, bias=False)
        # gamma(γ): γ=1-(α*o_gate+β*o_gate),  γ=o_gate*gamma_p, (α_p+β_p+γ_p) = o_gate^(-1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        # 组合重置输入
        input_gates = self.input_in(x) + self.hidden_in(hidden)
        i_gate, h_gate = input_gates.chunk(2, dim=-1)
        i_gate = torch.sigmoid(i_gate)  # 候选重置输入门
        h_gate = torch.tanh(h_gate)  # 候选隐藏状态
        ch_n = i_gate * h_gate  # 重置隐藏状态

        # 单独重置输入
        only_x_gate = self.only_x(x)
        x = torch.tanh(only_x_gate)
        only_h_gate = self.only_h(hidden)
        hidden = torch.tanh(only_h_gate)

        # 更新输出
        output_gates = self.input_out(x) + self.hidden_out(hidden) + self.candidate_hidden_out(ch_n)
        o_gate = torch.sigmoid(output_gates)  # 更新输出门
        alpha = self.alpha_p(o_gate)  # 更新单独的输入
        beta = self.beta_p(o_gate)  # 更新单独的过去隐藏状态
        gamma = 1 - alpha - beta  # 更新当前的候选隐藏状态
        h_next = alpha * x + beta * hidden + gamma * ch_n  # 最终隐藏状态输出
        return h_next

# 多层 ELTM
class ELTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(ELTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 创建多个ELTMCell，第一层特殊
        self.ELTMs = nn.ModuleList([ELTMcell(input_size, hidden_size)] +
                                   [ELTMcell(hidden_size, hidden_size) for _ in range(num_layers - 1)])

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        # x输出形状: [seq_len. batch_size, input_size]
        # state is hidden(h): [batch_size, hidden_size]
        seq_len, batch_size = x.shape[:2]

        # 需要初始化隐藏的第一层
        if state is None:
            hidden = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            (hidden) = state
            hidden = list(torch.unbind(hidden))

        # 收集每个时间步骤的输出
        out = []
        for t in range(seq_len):
            # 第一个时刻的输入是x本身
            inp = x[t]
            # 遍历 n 层
            for layer in range(self.num_layers):
                hidden[layer] = self.ELTMs[layer](inp, hidden[layer])
                # 将当前层的输出作为下一层的输入
                inp = hidden[layer]
            # 收集最后一刻的输出
            out.append(hidden[-1])

        # 将所有输出叠加在一起
        out = torch.stack(out)
        hidden = torch.stack(hidden)
        return out, hidden


if __name__ == "__main__":
    ELTM = ELTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    output, _ = ELTM(input)
    print(output.shape)  # ELTM输出维度torch.Size([5, 3, 20])
    # 如果要做实际预测任务，先取最后一时刻的输出，再使用一个Linear(hidden_size, output_size)
    # ELTM即插即用



