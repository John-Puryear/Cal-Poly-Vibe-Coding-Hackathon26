"""Pure tensor transition. No anatomy templates or target lookup at inference."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.nn.functional as F

CHANNELS = [
    "tissue", "anterior_fate", "eye_fate", "tail_fate",
    "voltage", "activator", "inhibitor", "stress",
    "fast_0", "fast_1", "fast_2", "fast_3",
    "regulatory", "polarity", "structure", "adaptation",
    "slow_0", "slow_1", "slow_2", "slow_3",
    "activity_trace", "consolidated_preference", "planaria_context", "face_context",
]

@dataclass
class Config:
    channels: int = 24
    hidden: int = 96
    message: int = 8
    rank: int = 4
    fire_rate: float = 0.5
    long_radius: int = 6
    long_gain: float = 0.10
    plasticity: bool = True
    consolidation: bool = True
    interaction: bool = True
    long_range: bool = True

    def __post_init__(self):
        if self.channels != 24 or self.hidden < 1 or self.message < 1 or self.rank < 1:
            raise ValueError("This demo uses 24 channels and positive layer widths")
        if not 0 < self.fire_rate <= 1 or not 0 <= self.long_gain <= 0.1:
            raise ValueError("fire_rate must be (0,1]; long_gain must be [0,0.1]")
        if self.long_radius < 2:
            raise ValueError("Long links must span at least two cells")

@dataclass
class TissueState:
    cells: torch.Tensor        # B,C,H,W
    edge_fast: torch.Tensor    # B,6,H,W; receiver-indexed directed edges
    edge_slow: torch.Tensor
    delayed: torch.Tensor      # B,2,M+2,H,W; message, voltage, source liveness

    def detach(self):
        return TissueState(*(x.detach() for x in self.tensors()))

    def tensors(self):
        return self.cells, self.edge_fast, self.edge_slow, self.delayed


def shift(x, dy, dx):
    """Receiver (y,x) reads source (y+dy,x+dx); zero exterior, never wraps."""
    h, w = x.shape[-2:]
    p = F.pad(x, (max(-dx, 0), max(dx, 0), max(-dy, 0), max(dy, 0)))
    return p[..., max(dy, 0):max(dy, 0)+h, max(dx, 0):max(dx, 0)+w]


class BDHNCA(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or Config()
        self.offsets = ((-1, 0), (1, 0), (0, -1), (0, 1),
                        (0, -c.long_radius), (0, c.long_radius))
        self.query = nn.Conv2d(24, c.message, 1)
        self.key = nn.Conv2d(24, c.message, 1)
        self.value = nn.Conv2d(24, c.message, 1)
        self.fast_factor = nn.Conv2d(4, c.rank, 1)
        self.slow_factor = nn.Conv2d(4, c.rank, 1)
        features = 5*24 + 2*c.message + c.rank
        self.update = nn.Sequential(nn.Conv2d(features, c.hidden, 1), nn.ReLU(),
                                    nn.Conv2d(c.hidden, 20, 1))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.register_buffer("rates", torch.tensor([.15]*4+[.30]*4+[.20]*4+[.015]*8)[None,:,None,None])

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def initial(self, cells):
        b, _, h, w = cells.shape
        z = cells.new_zeros(b, 6, h, w)
        return TissueState(cells, z, z.clone(),
                           cells.new_zeros(b, 2, self.config.message+2, h, w))

    def damage(self, state, keep):
        """Erase tissue and incident history while preserving remote memory."""
        keep=keep.to(state.cells.dtype)
        valid=torch.cat([keep*shift(keep,*o) for o in self.offsets],1)
        return TissueState(state.cells*keep,state.edge_fast*valid,state.edge_slow*valid,
                           state.delayed*valid[:,4:,None])

    def forward(self, state, control=None, fire=None):
        """One tick. control = voltage current [-1,1], gap block [0,1].

        A supplied B,1,H,W fire mask makes this function deterministic and portable.
        The 0.1 bound on long_gain is a feature/message amplitude budget, not a
        bound on every downstream neural sensitivity.
        """
        c = self.config
        x, ef, es, delayed = state.tensors()
        if control is None:
            control = x.new_zeros(x.shape[0], 2, *x.shape[-2:])
        current = control[:, :1].clamp(-1, 1)
        block = control[:, 1:2].clamp(0, 1)
        control = torch.cat((current, block), 1)
        if fire is None:
            fire = (torch.rand_like(x[:, :1]) < c.fire_rate).to(x.dtype)
        alive = (x[:, :1] > .1).to(x.dtype)
        frontier = (F.max_pool2d(x[:, :1], 3, 1, 1) > .1).to(x.dtype)
        q, k = torch.tanh(self.query(x)), torch.tanh(self.key(x))
        value = torch.tanh(self.value(x))
        neighbors = torch.stack([shift(x, *o) for o in self.offsets], 1)
        source_alive = (neighbors[:, :, :1] > .1).to(x.dtype)
        edge_alive = alive[:, None] * source_alive
        # Long edges additionally require that the delayed sender was alive.
        edge_alive = torch.cat((edge_alive[:, :4], edge_alive[:, 4:]*delayed[:, :, -1:]), 1)
        keys = torch.stack([shift(k, *o) for o in self.offsets], 1)
        scores = torch.sigmoid((q[:, None]*keys).sum(2)/(c.message**.5))
        coupling = (0.20 + .45*ef + .35*es) if c.plasticity else torch.full_like(ef, .20)
        weights = scores * coupling * edge_alive.squeeze(2)
        # Both sending and receiving tissue must have open conductance.
        source_open = torch.stack([shift(1-block, *o) for o in self.offsets], 1).squeeze(2)
        weights = weights * (1-block) * source_open
        local_w = weights[:, :4] / weights[:, :4].sum(1, keepdim=True).clamp_min(1.0)
        long_w = weights[:, 4:] / weights[:, 4:].sum(1, keepdim=True).clamp_min(1.0)
        if not c.long_range:
            long_w = long_w * 0
        near_v = torch.stack([shift(value, *o) for o in self.offsets[:4]], 1)
        local = (local_w[:, :, None]*near_v).sum(1)
        distant = c.long_gain * (long_w[:, :, None]*delayed[:, :, :c.message]).sum(1)
        # Fixed local perception, directional derivatives, and Laplacian.
        n, s, w, e = [neighbors[:, i] for i in range(4)]
        mean = (n+s+w+e)*.25
        gx, gy, lap = (e-w)*.5, (s-n)*.5, mean-x
        cross = torch.tanh(self.fast_factor(x[:,4:8]))*torch.tanh(self.slow_factor(x[:,12:16]))
        if not c.interaction:
            cross = cross*0
        # Intervention inputs affect physical current/conductance only. Keeping
        # them out of the learned rule avoids untrained direct fate shortcuts.
        features = torch.cat((x, mean, gx, gy, lap, local, distant, cross), 1)
        delta = torch.tanh(self.update(features)) * self.rates
        # Generic conductive transport and weak preference recall. No fate rule.
        local_dv = (local_w*(neighbors[:, :4, 4]-x[:, 4:5])).sum(1, keepdim=True)
        long_dv = (long_w*(delayed[:, :, c.message]-x[:, 4:5])).sum(1, keepdim=True)
        feedback = .015*x[:,21:22]*(1-x[:,15:16].clamp(0,1)) if c.consolidation else x[:,4:5]*0
        voltage_extra = .12*local_dv + .025*long_dv + .15*current + feedback
        extra = torch.cat((torch.zeros_like(x[:,:4]), voltage_extra, torch.zeros_like(x[:,5:20])), 1)
        dynamic = (x[:, :20] + fire*(delta+extra-.002*x[:, :20])).clamp(-1,1)
        # Visible fields and adaptation have nonnegative ranges.
        # Straight-through gradient for nonnegative fate bounds: a transient
        # negative proposal must not permanently kill a zero-initialized fate.
        visible = dynamic[:, :4] + (dynamic[:, :4].clamp(0,1)-dynamic[:, :4]).detach()
        dynamic = torch.cat((visible, dynamic[:,4:15],
                             dynamic[:,15:16].clamp(0,1), dynamic[:,16:20]), 1)
        # Repeated concordant activity transfers into a slower preference.
        concordance = (x[:,4:5]*x[:,12:13]).clamp(0,1)
        trace = x[:,20:21] + .04*fire*(x[:,4:5]-x[:,20:21])
        preference = x[:,21:22] + .003*fire*(concordance*trace-x[:,21:22])
        if not c.consolidation:
            trace, preference = trace*0, preference*0
        cells = torch.cat((dynamic, trace.clamp(-1,1), preference.clamp(-1,1), x[:,22:24]), 1)
        post = (F.max_pool2d(cells[:,:1],3,1,1) > .1).to(x.dtype)
        cells = cells * frontier * post
        # Edge reinforcement: agreement, finite headroom, adaptation, resource
        # competition, decay, and conservative fast-to-slow transfer.
        agreement = (1-(neighbors[:,:,4]-x[:,4:5]).abs()*.5).clamp(0,1)
        structural = (1-(neighbors[:,:,12]-x[:,12:13]).abs()*.5).clamp(0,1)
        eligible = agreement*structural*x[:,4:5].abs()*(1-x[:,15:16])
        headroom = (1-ef-es).clamp_min(0)
        compete = 1/(1+ef.sum(1,keepdim=True))
        transfer = .003*ef if c.consolidation else ef*0
        new_ef = (ef + .06*eligible*headroom*compete - .02*ef - transfer).clamp(0,1)
        new_es = (es + transfer - .0005*es).clamp(0,1)
        scale = (new_ef+new_es).clamp_min(1)
        new_ef, new_es = new_ef/scale, new_es/scale
        # No hidden edge memory survives deletion of either endpoint.
        new_alive = (cells[:,:1] > .1).to(x.dtype)
        valid = torch.cat([new_alive*shift(new_alive,*o) for o in self.offsets],1)
        enabled = x.new_tensor([1,1,1,1,int(c.long_range),int(c.long_range)])[None,:,None,None]
        valid = valid*enabled
        new_ef, new_es = new_ef*valid, new_es*valid
        if not c.plasticity:
            new_ef, new_es = new_ef*0, new_es*0
        if not c.consolidation:
            new_es = new_es*0
        packet = torch.cat((value, x[:,4:5], alive),1)
        new_delay = torch.stack([shift(packet,*o) for o in self.offsets[4:]],1)
        new_delay = new_delay*new_alive[:,None]*enabled[:,4:,None]
        return TissueState(cells, new_ef, new_es, new_delay)

    def metadata(self):
        return {"format":"bdh-nca-v2", "config":asdict(self.config), "channels":CHANNELS,
                "parameter_count":self.parameter_count, "offsets":self.offsets}
