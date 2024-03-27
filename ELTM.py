import torch
from torch import nn
from typing import Optional

# Efficient Long Term Memory: ELTM
class ELTMcell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ELTMcell, self).__init__()
        # Reset the input weight matrix initialization
        self.input_in = nn.Linear(input_size, hidden_size * 2, bias=False)
        self.hidden_in = nn.Linear(hidden_size, hidden_size * 2)
        self.only_x = nn.Linear(input_size, hidden_size, bias=False)
        self.only_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Update the output weight matrix initialization
        self.input_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.candidate_hidden_out = nn.Linear(hidden_size, hidden_size)
        # Branch output parameter initialization
        self.alpha_p = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_p = nn.Linear(hidden_size, hidden_size, bias=False)
        # gamma(γ): γ=1-(α*o_gate+β*o_gate),  γ=o_gate*gamma_p, (α_p+β_p+γ_p) = o_gate^(-1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        # Combined reset input
        input_gates = self.input_in(x) + self.hidden_in(hidden)
        i_gate, h_gate = input_gates.chunk(2, dim=-1)
        i_gate = torch.sigmoid(i_gate)  # Candidate reset input gate
        h_gate = torch.tanh(h_gate)  # Candidate hidden state
        ch_n = i_gate * h_gate  # Reset hidden state

        # Separate reset input
        only_x_gate = self.only_x(x)
        x = torch.tanh(only_x_gate)
        only_h_gate = self.only_h(hidden)
        hidden = torch.tanh(only_h_gate)

        # Update output
        output_gates = self.input_out(x) + self.hidden_out(hidden) + self.candidate_hidden_out(ch_n)
        o_gate = torch.sigmoid(output_gates)  # Update output gate
        alpha = self.alpha_p(o_gate)  # Update individual inputs
        beta = self.beta_p(o_gate)  # Update separate past hidden states
        gamma = 1 - alpha - beta  # Update the current candidate hidden state
        h_next = alpha * x + beta * hidden + gamma * ch_n  # The final hidden state output

        return h_next

# multilayer ELTM
class ELTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(ELTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Create multiple EltMcells, with the first layer special
        self.ELTMs = nn.ModuleList([ELTMcell(input_size, hidden_size)] +
                                   [ELTMcell(hidden_size, hidden_size) for _ in range(num_layers - 1)])

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        # Output shape of x: [seq_len. batch_size, input_size]
        # state is hidden(h): [batch_size, hidden_size]
        seq_len, batch_size = x.shape[:2]

        # Collect the output for each time step
        if state is None:
            hidden = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            (hidden) = state
            hidden = list(torch.unbind(hidden))

        # Collect the output for each time step
        out = []
        for t in range(seq_len):
            # The input at the first moment is x itself
            inp = x[t]
            # Traverse n layers
            for layer in range(self.num_layers):
                hidden[layer] = self.ELTMs[layer](inp, hidden[layer])
                # Take the output of the current layer as the input of the next layer
                inp = hidden[layer]
            # Collect the last minute output
            out.append(hidden[-1])

        # Overlay all outputs together
        out = torch.stack(out)
        hidden = torch.stack(hidden)
        return out, hidden

if __name__ == "__main__":
    ELTM = ELTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    output, _ = ELTM(input)
    print(output.shape)  # ELTM Output dimension: torch.Size([5, 3, 20])
    # If you want to do an actual prediction task, take the output at the last moment and use one Linear(hidden_size, output_size)
    # ELTM plugs and plays



