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
    control_sensing: bool = False
    induction_memory: bool = False
    loop_reinforcement: bool = False
    morphogenetic_readout: bool = False
    explicit_induction_state: bool = False
    distinct_cascades: bool = False
    compositional_eye_recruitment: bool = False
    injury_provenance: bool = False
    temporal_face_gate: bool = False
    axial_default_head: bool = False
    axial_head_threshold: float = .15
    octanol_voltage_repatterning: bool = False

    def __post_init__(self):
        if self.channels != 24 or self.hidden < 1 or self.message < 1 or self.rank < 1:
            raise ValueError("This demo uses 24 channels and positive layer widths")
        if not 0 < self.fire_rate <= 1 or not 0 <= self.long_gain <= 0.1:
            raise ValueError("fire_rate must be (0,1]; long_gain must be [0,0.1]")
        if self.long_radius < 2:
            raise ValueError("Long links must span at least two cells")
        if not 0 <= self.axial_head_threshold < 1:
            raise ValueError("axial_head_threshold must be in [0,1)")

@dataclass
class TissueState:
    cells: torch.Tensor        # B,C,H,W
    edge_fast: torch.Tensor    # B,6,H,W; receiver-indexed directed edges
    edge_slow: torch.Tensor
    delayed: torch.Tensor      # B,2,M+2,H,W; message, voltage, source liveness
    induction: torch.Tensor | None = None  # optional B,2,H,W provenance memory

    def detach(self):
        return TissueState(*(x.detach() for x in self.tensors()))

    def tensors(self):
        core = (self.cells, self.edge_fast, self.edge_slow, self.delayed)
        return core if self.induction is None else (*core, self.induction)


def shift(x, dy, dx):
    """Receiver (y,x) reads source (y+dy,x+dx); zero exterior, never wraps."""
    h, w = x.shape[-2:]
    p = F.pad(x, (max(-dx, 0), max(dx, 0), max(-dy, 0), max(dy, 0)))
    return p[..., max(dy, 0):max(dy, 0)+h, max(dx, 0):max(dx, 0)+w]


class BDHNCA(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or Config()
        if c.explicit_induction_state and not c.morphogenetic_readout:
            raise ValueError("Explicit induction state requires the shared morphogenetic readout")
        if c.distinct_cascades and not c.explicit_induction_state:
            raise ValueError("Distinct cascades require explicit induction state")
        if c.injury_provenance and not c.explicit_induction_state:
            raise ValueError("Injury provenance requires explicit induction state")
        if c.axial_default_head and not c.injury_provenance:
            raise ValueError("Axial default-head circuit requires injury provenance")
        if c.octanol_voltage_repatterning and not c.axial_default_head:
            raise ValueError("Octanol voltage repatterning requires axial default-head circuit")
        self.offsets = ((-1, 0), (1, 0), (0, -1), (0, 1),
                        (0, -c.long_radius), (0, c.long_radius))
        self.query = nn.Conv2d(24, c.message, 1)
        self.key = nn.Conv2d(24, c.message, 1)
        self.value = nn.Conv2d(24, c.message, 1)
        self.fast_factor = nn.Conv2d(4, c.rank, 1)
        self.slow_factor = nn.Conv2d(4, c.rank, 1)
        features = 5*24 + 2*c.message + c.rank + (3 if c.control_sensing else 0) + (2 if c.explicit_induction_state else 0)
        self.update = nn.Sequential(nn.Conv2d(features, c.hidden, 1), nn.ReLU(),
                                    nn.Conv2d(c.hidden, 20, 1))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        if c.induction_memory:
            # An intervention-gated local writer to two slow latent channels.
            # The shared anatomy rule above still reads the full state and
            # determines every visible field; the writer supplies no anatomy.
            self.memory_writer = nn.Sequential(nn.Conv2d(11, 32, 1), nn.ReLU(),
                                               nn.Conv2d(32, 2, 1))
            nn.init.zeros_(self.memory_writer[-1].weight)
            nn.init.zeros_(self.memory_writer[-1].bias)
        if c.loop_reinforcement:
            if not c.induction_memory:
                raise ValueError("Loop reinforcement requires induction memory")
            # Zero is an exact function-preserving upgrade. Training determines
            # whether latent-memory concordance deserves reciprocal edge gain.
            self.loop_gain = nn.Parameter(torch.zeros(()))
        if c.morphogenetic_readout:
            if not c.induction_memory or not c.loop_reinforcement:
                raise ValueError("Morphogenetic readout requires memory and BDH loops")
            # Shared edge-conditioned route to visible, electrical, regulatory,
            # polarity, and structural dynamics. No fixed phenotype template.
            self.morph_readout = nn.Sequential(nn.Conv2d(54+(2 if c.explicit_induction_state else 0), 64, 1), nn.ReLU(),
                                               nn.Conv2d(64, 7, 1))
            nn.init.zeros_(self.morph_readout[-1].weight)
            nn.init.zeros_(self.morph_readout[-1].bias)
            if c.distinct_cascades:
                # The body-boundary channel is absent from both outputs. The
                # head and eye protocols have different state inputs and gates.
                self.head_cascade = nn.Sequential(nn.Conv2d(27, 48, 1), nn.ReLU(), nn.Conv2d(48, 4, 1))
                self.eye_cascade = nn.Sequential(nn.Conv2d(27, 48, 1), nn.ReLU(), nn.Conv2d(48, 3, 1))
                for branch in (self.head_cascade,self.eye_cascade):
                    nn.init.zeros_(branch[-1].weight); nn.init.zeros_(branch[-1].bias)
        self.register_buffer("rates", torch.tensor([.15]*4+[.30]*4+[.20]*4+[.015]*8)[None,:,None,None])

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def initial(self, cells):
        b, _, h, w = cells.shape
        z = cells.new_zeros(b, 6, h, w)
        induction = cells.new_zeros(b,3 if self.config.injury_provenance else 2,h,w) if self.config.explicit_induction_state else None
        return TissueState(cells, z, z.clone(),
                           cells.new_zeros(b, 2, self.config.message+2, h, w), induction)

    def damage(self, state, keep):
        """Erase tissue and incident history while preserving remote memory."""
        keep=keep.to(state.cells.dtype)
        valid=torch.cat([keep*shift(keep,*o) for o in self.offsets],1)
        cells = state.cells*keep
        if self.config.induction_memory:
            # A cut exposes surviving tissue to a transient injury signal.
            # This is a synthetic physical cue, not a head or eye label.
            lost = (1-keep)*(state.cells[:,:1]>.1)
            border = keep*F.max_pool2d(lost,3,1,1)*(state.cells[:,:1]>.1)
            stress = torch.maximum(cells[:,7:8], .8*border)
            cells = torch.cat((cells[:,:7], stress, cells[:,8:]),1)
        induction = state.induction*keep if state.induction is not None else None
        if self.config.injury_provenance:
            wound_region=(F.max_pool2d(border,7,1,3)*keep) if self.config.axial_default_head else border
            injury=torch.maximum(induction[:,2:3],wound_region)
            induction=torch.cat((induction[:,:2],injury),1)
        return TissueState(cells,state.edge_fast*valid,state.edge_slow*valid,
                           state.delayed*valid[:,4:,None],induction)

    def forward(self, state, control=None, fire=None):
        """One tick. control = current, broad coupling block, optional long-only block.

        A supplied B,1,H,W fire mask makes this function deterministic and portable.
        The 0.1 bound on long_gain is a feature/message amplitude budget, not a
        bound on every downstream neural sensitivity.
        """
        c = self.config
        x, ef, es, delayed = state.cells, state.edge_fast, state.edge_slow, state.delayed
        induction = state.induction
        if c.explicit_induction_state and induction is None:
            raise ValueError("Explicit induction checkpoint requires the fifth recurrent state tensor")
        if control is None:
            control = x.new_zeros(x.shape[0], 2, *x.shape[-2:])
        current = control[:, :1].clamp(-1, 1)
        block = control[:, 1:2].clamp(0, 1)
        long_block = control[:,2:3].clamp(0,1) if control.shape[1]>2 else torch.zeros_like(block)
        purine_input = control[:,3:4].clamp(0,1) if control.shape[1]>3 else torch.zeros_like(block)
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
        long_open=torch.stack([shift(1-long_block,*o) for o in self.offsets[4:]],1).squeeze(2)
        weights=torch.cat((weights[:,:4],weights[:,4:]*(1-long_block)*long_open),1)
        local_w = weights[:, :4] / weights[:, :4].sum(1, keepdim=True).clamp_min(1.0)
        long_w = weights[:, 4:] / weights[:, 4:].sum(1, keepdim=True).clamp_min(1.0)
        if not c.long_range:
            long_w = long_w * 0
        near_v = torch.stack([shift(value, *o) for o in self.offsets[:4]], 1)
        local = (local_w[:, :, None]*near_v).sum(1)
        distant = c.long_gain * (long_w[:, :, None]*delayed[:, :, :c.message]).sum(1)
        # Fixed local perception, directional derivatives, and Laplacian.
        sensed=neighbors[:,:4].clone()
        electrical_block=1-(1-block)*source_open[:,:4]
        sensed[:,:,4:8]=sensed[:,:,4:8]-electrical_block[:,:,None]*(sensed[:,:,4:8]-x[:,None,4:8])
        n, s, w, e = [sensed[:, i] for i in range(4)]
        mean = (n+s+w+e)*.25
        gx, gy, lap = (e-w)*.5, (s-n)*.5, mean-x
        cross = torch.tanh(self.fast_factor(x[:,4:8]))*torch.tanh(self.slow_factor(x[:,12:16]))
        if not c.interaction:
            cross = cross*0
        features = torch.cat((x, mean, gx, gy, lap, local, distant, cross), 1)
        if c.control_sensing:
            # Optional, trainable awareness of physical current/conductance.
            # This is a synthetic assumption; anatomy is never supplied here.
            features = torch.cat((features, current, block, long_block), 1)
        if c.explicit_induction_state:
            features = torch.cat((features, induction[:,:2]),1)
        delta = torch.tanh(self.update(features)) * self.rates
        if c.temporal_face_gate:
            # A face-specific developmental dependency: learned voltage must
            # appear before visible/head-eye, regulatory and structure updates.
            # The upstream physical cue is outside these gated fields.
            ready=((x[:,4:5].abs()-.12)/.32).clamp(0,1)
            gate_face=1-x[:,23:24].clamp(0,1)*(1-ready)
            delta=torch.cat((delta[:,:1],delta[:,1:3]*gate_face,delta[:,3:12],
                             delta[:,12:13]*gate_face,delta[:,13:14],
                             delta[:,14:15]*gate_face,delta[:,15:]),1)
        # Generic conductive transport and weak preference recall. No fate rule.
        local_dv = (local_w*(neighbors[:, :4, 4]-x[:, 4:5])).sum(1, keepdim=True)
        long_dv = (long_w*(delayed[:, :, c.message]-x[:, 4:5])).sum(1, keepdim=True)
        feedback = .015*x[:,21:22]*(1-x[:,15:16].clamp(0,1)) if c.consolidation else x[:,4:5]*0
        voltage_extra = .12*local_dv + .025*long_dv + .15*current + feedback
        if c.octanol_voltage_repatterning:
            # Synthetic GJ/voltage coupling hypothesis: loss of communication
            # depolarizes injured posterior-facing tissue; consolidated local
            # organizer memory then biases voltage after the block is washed out.
            # No intervention, injury, or supra-threshold memory -> no new drive.
            deprived = block*induction[:,2:3]*(-x[:,13:14]).clamp(0,1)
            # Voltage recall has a lower threshold than visible head induction:
            # a submorphological field can persist and later guide anatomy.
            recalled = (induction[:,0:1]-.10).clamp(0,1)
            voltage_extra = voltage_extra + (.06*deprived + .35*recalled)*(.55-x[:,4:5])
        extra = torch.cat((torch.zeros_like(x[:,:4]), voltage_extra, torch.zeros_like(x[:,5:20])), 1)
        dynamic = (x[:, :20] + fire*(delta+extra-.002*x[:, :20])).clamp(-1,1)
        if c.induction_memory:
            purine_stimulus = purine_input if c.explicit_induction_state else (x[:,8:9]-.5).clamp(0,1)*2
            gate = torch.maximum(torch.maximum(block,long_block),
                                 torch.maximum(current.abs(),purine_stimulus))
            writer_input = torch.cat((x[:,0:1],x[:,4:9],x[:,12:14],
                                      current,block,long_block),1)
            writer_drive=torch.tanh(self.memory_writer(writer_input))
            if c.injury_provenance:
                head_stimulus=torch.maximum(block,long_block)*induction[:,2:3]
                eye_stimulus=torch.maximum(current.abs(),purine_stimulus)
                memory_delta=.02*writer_drive*torch.cat((head_stimulus,eye_stimulus),1)*fire
                if c.axial_default_head:
                    # The normal posterior electrical axis suppresses a default
                    # head organizer. Prolonged local GJ blockade lifts that
                    # suppression only in injured posterior-facing tissue.
                    released=.025*block*induction[:,2:3]*(-x[:,13:14]).clamp(0,1)*fire
                    memory_delta=torch.cat((memory_delta[:,:1]+released,memory_delta[:,1:2]),1)
            else:
                memory_delta = .02*writer_drive*gate*fire
            if c.explicit_induction_state:
                # Two stable regimes: zero is invariant without treatment;
                # supra-threshold memory consolidates after washout.
                memory_state=induction[:,:2]
                bistable = .01*memory_state*(1-memory_state)*(memory_state-.25)*fire
                proposed_induction = memory_state+memory_delta+bistable
                bounded=proposed_induction + (proposed_induction.clamp(0,1)-proposed_induction).detach()
                induction_next=torch.cat((bounded,induction[:,2:3]*.995),1) if c.injury_provenance else bounded
            else:
                dynamic = torch.cat((dynamic[:,:16],
                                     (dynamic[:,16:18]+memory_delta).clamp(-1,1),
                                     dynamic[:,18:20]),1)
        if c.morphogenetic_readout:
            source_memory = induction[:,:2] if c.explicit_induction_state else x[:,16:18]
            latent_gate = torch.maximum((source_memory[:,0:1]-.15).clamp_min(0),
                                        (source_memory[:,1:2]-.02).clamp_min(0)).clamp(0,1)
            edge_gate = (4*(ef[:,:4]+es[:,:4]).mean(1,keepdim=True)).clamp(0,1)
            morph_features = torch.cat((x,mean,ef+es,induction[:,:2]),1) if c.explicit_induction_state else torch.cat((x,mean,ef+es),1)
            morph_rate = .16 if c.explicit_induction_state else .08
            if c.distinct_cascades:
                edge_features=(ef+es)[:,:4]
                head_ix=(0,1,3,4,7,12,13,18,20,22)
                eye_ix=(0,2,4,5,8,12,14,19,21,23)
                head_features=torch.cat((x[:,head_ix],mean[:,head_ix],edge_features,
                                         induction[:,0:1],x[:,7:8],x[:,13:14]),1)
                eye_features=torch.cat((x[:,eye_ix],mean[:,eye_ix],edge_features,
                                        induction[:,1:2],x[:,5:6],x[:,8:9]),1)
                head_threshold=c.axial_head_threshold
                head_gate=(induction[:,0:1]-head_threshold).clamp(0,1)*edge_gate*fire
                eye_signal=(induction[:,1:2]-.02).clamp(0,1)
                if c.compositional_eye_recruitment:
                    # A locally induced head can call the same eye protocol.
                    # The branch still determines eye position from local state.
                    head_organizer=(induction[:,0:1]-head_threshold).clamp(0,1)*dynamic[:,1:2].clamp(0,1)
                    eye_signal=torch.maximum(eye_signal, .8*head_organizer)
                eye_gate=eye_signal*edge_gate*fire
                head_delta=morph_rate*torch.tanh(self.head_cascade(head_features))*head_gate
                eye_delta=morph_rate*torch.tanh(self.eye_cascade(eye_features))*eye_gate
                dynamic=torch.cat((dynamic[:,:1],
                    (dynamic[:,1:2]+head_delta[:,:1]).clamp(-1,1),
                    (dynamic[:,2:3]+eye_delta[:,:1]).clamp(-1,1),
                    (dynamic[:,3:4]+head_delta[:,1:2]).clamp(-1,1),
                    (dynamic[:,4:5]+head_delta[:,2:3]).clamp(-1,1),dynamic[:,5:12],
                    (dynamic[:,12:13]+eye_delta[:,1:2]).clamp(-1,1),
                    (dynamic[:,13:14]+head_delta[:,3:4]).clamp(-1,1),
                    (dynamic[:,14:15]+eye_delta[:,2:3]).clamp(-1,1),dynamic[:,15:20]),1)
            else:
                morph_delta = morph_rate*torch.tanh(self.morph_readout(morph_features))*latent_gate*edge_gate*fire
                dynamic = torch.cat((dynamic[:,:1],
                                     (dynamic[:,1:4]+morph_delta[:,:3]).clamp(-1,1),
                                     (dynamic[:,4:5]+morph_delta[:,3:4]).clamp(-1,1),
                                     dynamic[:,5:12],
                                     (dynamic[:,12:15]+morph_delta[:,4:7]).clamp(-1,1),
                                     dynamic[:,15:20]),1)
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
        if c.loop_reinforcement:
            source_memory = induction[:,:2] if c.explicit_induction_state else x[:,16:18]
            memory = source_memory.clamp_min(0).sum(1,keepdim=True).clamp(0,1)
            neighbor_memory = torch.stack([shift(memory,*o) for o in self.offsets],1).squeeze(2)
            reciprocal = torch.stack((shift((ef+es)[:,1:2],-1,0),
                                      shift((ef+es)[:,0:1],1,0),
                                      shift((ef+es)[:,3:4],0,-1),
                                      shift((ef+es)[:,2:3],0,1),
                                      shift((ef+es)[:,5:6],0,-c.long_radius),
                                      shift((ef+es)[:,4:5],0,c.long_radius)),1).squeeze(2)
            loop_drive = memory*neighbor_memory*agreement*structural*(1-x[:,15:16])
            eligible = (eligible + torch.tanh(self.loop_gain.clamp_min(0))*loop_drive*(.2+.8*reciprocal)).clamp_min(0)
        eligibility_open=(1-block)*source_open
        eligibility_open=torch.cat((eligibility_open[:,:4],eligibility_open[:,4:]*(1-long_block)*long_open),1)
        eligible=eligible*eligibility_open
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
        if c.explicit_induction_state:
            induction_next = induction_next * frontier * post
        else:
            induction_next = None
        return TissueState(cells, new_ef, new_es, new_delay, induction_next)

    def metadata(self):
        return {"format":"bdh-nca-v2", "config":asdict(self.config), "channels":CHANNELS,
                "parameter_count":self.parameter_count, "offsets":self.offsets,
                "control_channels":["voltage_current","broad_coupling_block","optional_long_only_block","optional_purine_stimulus"],
                "blockade_semantics":"electrical-perception-and-eligibility-v2"}
