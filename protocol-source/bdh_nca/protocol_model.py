"""Small hybrid BDH/NCA protocol model for synthetic regeneration experiments.

The transition uses local tissue, electrical, biochemical, regulatory, and
structural fields. Its spatial rules are inductive biases, not measurements of
planarian kinetics. No target image or training teacher is read at inference.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import CHANNELS, shift


@dataclass
class ProtocolConfig:
    channels: int = 24
    fire_rate: float = .5
    long_radius: int = 6
    hidden: int = 16
    protocol_version: int = 1

    def __post_init__(self):
        if self.channels != 24 or self.hidden < 1 or not 0 < self.fire_rate <= 1:
            raise ValueError("Expected 24 channels, a positive hidden width, and valid fire rate")


@dataclass
class ProtocolState:
    cells: torch.Tensor          # B,24,H,W
    edge_fast: torch.Tensor      # B,6,H,W
    edge_slow: torch.Tensor      # B,6,H,W
    delayed: torch.Tensor        # B,2,10,H,W
    induction: torch.Tensor      # B,3,H,W: head memory, eye memory, injury

    def tensors(self):
        return self.cells, self.edge_fast, self.edge_slow, self.delayed, self.induction

    def detach(self):
        return ProtocolState(*(v.detach() for v in self.tensors()))


class ProtocolNCA(nn.Module):
    """A tissue-intrinsic protocol with learnable rates and a local residual.

    Broad blockade affects local conductance and can consolidate a posterior
    organizer only after a cut. Short or long-only blockade does not cross the
    organizer's bistable threshold at the default parameter values. Heads use
    one common organ program; eyes are placed as two lateral foci per head.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ProtocolConfig()
        self.exposure_log_gain = nn.Parameter(torch.zeros(()))
        self.memory_threshold_delta = nn.Parameter(torch.zeros(()))
        self.voltage_log_gain = nn.Parameter(torch.zeros(()))
        self.head_log_gain = nn.Parameter(torch.zeros(()))
        self.eye_log_gain = nn.Parameter(torch.zeros(()))
        self.tissue_log_gain = nn.Parameter(torch.zeros(()))
        self.eye_axial_bias = nn.Parameter(torch.zeros(()))
        self.eye_lateral_bias = nn.Parameter(torch.zeros(()))
        self.loop_gain = nn.Parameter(torch.tensor(.5))
        # A small shared trainable correction is deliberately rate limited. Its
        # inputs are local physical fields, never a phenotype label or target.
        self.residual = nn.Sequential(
            nn.Conv2d(9, self.config.hidden, 1), nn.Tanh(),
            nn.Conv2d(self.config.hidden, 5, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def metadata(self):
        return {"format": "bdh-protocol-v1", "config": asdict(self.config),
                "channels": CHANNELS, "parameter_count": self.parameter_count,
                "control_channels": ["voltage_current", "broad_coupling_block",
                                     "optional_long_only_block", "optional_purine_stimulus"],
                "synthetic_mechanism": True}

    def initial(self, cells):
        if cells.ndim != 4 or cells.shape[1] != 24:
            raise ValueError("initial cells must have shape B,24,H,W")
        b, _, h, w = cells.shape
        # A coarse five-number body prior is estimated from the supplied
        # physical tissue, then stored redundantly in surviving cells. This is
        # a synthetic structural-memory assumption, not an intact-body target
        # image. A cut removes the prior from deleted tissue along with cells.
        tissue = cells[:, :1].clamp(0, 1)
        ys, xs = torch.meshgrid(torch.arange(h, device=cells.device, dtype=cells.dtype),
                                torch.arange(w, device=cells.device, dtype=cells.dtype),
                                indexing="ij")
        xs, ys = xs[None,None], ys[None,None]
        mass = tissue.sum((-2,-1),keepdim=True).clamp_min(1e-4)
        cx = (tissue*xs).sum((-2,-1),keepdim=True)/mass
        cy = (tissue*ys).sum((-2,-1),keepdim=True)/mass
        vx = (tissue*(xs-cx).square()).sum((-2,-1),keepdim=True)/mass
        vy = (tissue*(ys-cy).square()).sum((-2,-1),keepdim=True)/mass
        cov = (tissue*(xs-cx)*(ys-cy)).sum((-2,-1),keepdim=True)/mass
        angle = .5*torch.atan2(2*cov,vx-vy)
        discr = torch.sqrt(.25*(vx-vy).square()+cov.square()+1e-6)
        major = 2*torch.sqrt((.5*(vx+vy)+discr).clamp_min(.01))
        minor = 2*torch.sqrt((.5*(vx+vy)-discr).clamp_min(.01))
        code = torch.cat((2*cx/max(w-1,1)-1,2*cy/max(h-1,1)-1,
                          major/w,minor/h,angle/torch.pi),1)
        live = (tissue>.1).to(cells.dtype)
        cells = cells.clone()
        cells[:,9:12] = code[:,:3]*live
        cells[:,16:18] = code[:,3:5]*live
        return ProtocolState(cells,
                             cells.new_zeros(b, 6, h, w),
                             cells.new_zeros(b, 6, h, w),
                             cells.new_zeros(b, 2, 10, h, w),
                             cells.new_zeros(b, 3, h, w))

    def damage(self, state, keep):
        """Remove cut tissue and incident edges; retain only spared memory."""
        keep = keep.to(state.cells.dtype)
        old_live = (state.cells[:, :1] > .1).to(keep.dtype)
        lost = old_live * (1 - keep)
        border = keep * old_live * F.max_pool2d(lost, 3, 1, 1)
        cells = state.cells * keep
        cells = torch.cat((cells[:, :7], torch.maximum(cells[:, 7:8], .8 * border),
                           torch.maximum(cells[:, 8:9], border), cells[:, 9:]), 1)
        # Injury extends a few cells into spared tissue, so the middle-band
        # exposure can be sensed while only the cut ends are physically open.
        injury = torch.maximum(state.induction[:, 2:3] * keep,
                               F.max_pool2d(border, 7, 1, 3) * keep)
        induction = torch.cat((state.induction[:, :2] * keep, injury), 1)
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1),
                   (0, -self.config.long_radius), (0, self.config.long_radius))
        valid = torch.cat([keep * shift(keep, *o) for o in offsets], 1)
        return ProtocolState(cells, state.edge_fast * valid,
                             state.edge_slow * valid,
                             state.delayed * valid[:, 4:, None], induction)

    def forward(self, state, control=None, fire=None):
        x, ef, es, delayed, ind = state.tensors()
        b, _, h, w = x.shape
        if control is None:
            control = x.new_zeros(b, 4, h, w)
        current = control[:, :1].clamp(-1, 1)
        block = control[:, 1:2].clamp(0, 1)
        long_block = control[:, 2:3].clamp(0, 1) if control.shape[1] > 2 else torch.zeros_like(block)
        purine = control[:, 3:4].clamp(0, 1) if control.shape[1] > 3 else torch.zeros_like(block)
        if fire is None:
            fire = (torch.rand_like(x[:, :1]) < self.config.fire_rate).to(x.dtype)
        body, head, eye, tail = [x[:, i:i+1] for i in range(4)]
        voltage, activator, inhibitor = [x[:, i:i+1] for i in (4, 5, 6)]
        physical, repair_budget = x[:, 7:8], x[:, 8:9]
        regulatory, polarity, structure = [x[:, i:i+1] for i in (12, 13, 14)]
        face = x[:, 23:24].clamp(0, 1)
        head_memory, eye_memory, injury = [ind[:, i:i+1] for i in range(3)]
        alive = (body > .1).to(x.dtype)
        nearby = F.max_pool2d(body, 3, 1, 1)
        near_budget = F.max_pool2d(repair_budget, 3, 1, 1)

        code_field = torch.cat((x[:,9:12],x[:,16:18]),1)
        code = (code_field*alive).sum((-2,-1),keepdim=True)/alive.sum((-2,-1),keepdim=True).clamp_min(1)
        cx = (code[:,0:1]+1)*max(w-1,1)/2
        cy = (code[:,1:2]+1)*max(h-1,1)/2
        major = (code[:,2:3]*w).clamp_min(1)
        minor = (code[:,3:4]*h).clamp_min(1)
        angle = code[:,4:5]*torch.pi
        ys_grid, xs_grid = torch.meshgrid(torch.arange(h,device=x.device,dtype=x.dtype),
                                         torch.arange(w,device=x.device,dtype=x.dtype),indexing="ij")
        xs_grid, ys_grid = xs_grid[None,None],ys_grid[None,None]
        axial = (xs_grid-cx)*angle.cos()+(ys_grid-cy)*angle.sin()
        lateral_axis = -(xs_grid-cx)*angle.sin()+(ys_grid-cy)*angle.cos()
        allowed = ((axial/major).square()+(lateral_axis/minor).square() <= 1).to(x.dtype)

        # Wound budget spreads through only the fitted physical-body envelope.
        # New tissue must touch surviving tissue; it cannot nucleate an island.
        growth_gate = ((near_budget - .04) * 9).clamp(0, 1)
        growth = (.28 * self.tissue_log_gain.exp().clamp(.5, 2) * growth_gate
                  * (nearby - body).clamp_min(0) * allowed)
        new_body = (body + fire * (growth + .035 * alive * (1 - body))).clamp(0, 1)
        newborn = ((body <= .1) & (new_body > .1)).to(x.dtype)
        new_budget = torch.maximum(repair_budget * .97,
                                   .85 * near_budget * newborn).clamp(0, 1)
        support = F.avg_pool2d(alive, 3, 1, 1).clamp_min(1e-4)
        inherited_polarity = F.avg_pool2d(polarity * alive, 3, 1, 1) / support
        inherited_physical = F.avg_pool2d(physical * alive, 3, 1, 1) / support
        polarity = polarity * (1 - newborn) + inherited_polarity * newborn
        physical = physical * (1 - newborn) + inherited_physical * newborn
        face = face * (1 - newborn) + F.max_pool2d(face, 3, 1, 1) * newborn
        live = (new_body > .1).to(x.dtype)
        code_next = code.expand(b,5,h,w)*live

        # Reversible coupling loss exposes a local, wound-conditioned electrical
        # tendency before the posterior organizer consolidates. The spatial
        # spread is a synthetic assumption, not an injected voltage pulse.
        # Cubic dose response keeps a half-strength block a negative control.
        posterior = torch.sigmoid(12 * (-polarity - .12)) * (1 - face)
        memory_rate = .025 * self.exposure_log_gain.exp().clamp(.4, 2)
        threshold = .38 + .10 * torch.tanh(self.memory_threshold_delta)
        blocked_injury = block.pow(3) * injury
        coupling_signal = F.max_pool2d(blocked_injury, 13, 1, 6) * posterior.sqrt()
        # The previous tick's voltage must enter the induction window before
        # blocked wound tissue can build a persistent head organizer.
        voltage_window = torch.sigmoid(24 * (x[:, 4:5] + .02))
        exposure = memory_rate * coupling_signal * voltage_window * (1 - head_memory)
        bistable = .025 * head_memory * (1 - head_memory) * (head_memory - threshold)
        head_memory = (head_memory + fire * (exposure + bistable)).clamp(0, 1) * live
        # Purine/voltage-window exercise remains separate from axial head memory.
        eye_window = torch.exp(-((voltage + .35) / .18).square())
        eye_memory = (eye_memory + fire * (.012 * purine * eye_window * (1-eye_memory)
                                            + .012 * eye_memory * (1-eye_memory) * (eye_memory-.30))).clamp(0, 1) * live
        injury = (injury * .995).clamp(0, 1) * live
        nearby_head_memory = F.max_pool2d(head_memory, 9, 1, 4)
        posterior_program = torch.sigmoid(28 * (nearby_head_memory - threshold - .06)) * posterior.sqrt()

        # Conductance and local voltage diffusion are explicitly physical.
        local_voltage = F.avg_pool2d(voltage * live, 3, 1, 1) / F.avg_pool2d(live, 3, 1, 1).clamp_min(1e-4)
        planarian_voltage = (.62 * polarity + .78 * posterior_program
                             + .85 * coupling_signal - .50 * eye)
        xs = torch.arange(w, device=x.device, dtype=x.dtype)[None, None, None, :]
        row_mass = new_body.sum(-1, keepdim=True).clamp_min(1e-4)
        center_x = (new_body * xs).sum(-1, keepdim=True) / row_mass
        relative_x = xs - center_x
        face_bilateral = torch.exp(-.5 * ((relative_x.abs()-3.5)/.9).square())
        broad_face = (torch.exp(-.5*((physical-.065)/.095).square())
                      * torch.exp(-.5*(relative_x/7).square()))
        electric_eye_pair = (face_bilateral
                             * torch.exp(-.5*((physical-.07)/.024).square()))
        electric_mouth = (torch.exp(-.5*((physical+.10)/.03).square())
                          * torch.exp(-.5*(relative_x/4).square()))
        # Voltage foci form from the physical cue before visible eyes appear.
        # These Gaussian response windows are synthetic protocol assumptions.
        face_voltage = .58*broad_face - .75*electric_eye_pair - .45*electric_mouth
        voltage_target = (1-face) * planarian_voltage + face * face_voltage
        voltage_target = voltage_target + .15 * current
        diffusion = .07 * (1-block) * (local_voltage - voltage)
        packet_alive = delayed[:, :, 9].sum(1, keepdim=True)
        long_voltage = delayed[:, :, 8].sum(1, keepdim=True) / packet_alive.clamp_min(1)
        long_diff = .025 * (1-block) * (1-long_block) * (packet_alive>0).to(x.dtype) * (long_voltage-voltage)
        edge_recall = .02 * torch.tanh(self.loop_gain) * es[:, :4].mean(1, keepdim=True) * head_memory
        voltage = (voltage + fire * (.25 * self.voltage_log_gain.exp().clamp(.5, 2)
                                    * (voltage_target - voltage) + diffusion + long_diff + edge_recall)).clamp(-1, 1) * live

        # Face voltage must appear before the regulatory and visible protocol.
        # Read the *previous* tick's voltage so an external voltage clamp can
        # prevent downstream fates instead of leaking through same-tick Vmem.
        # A small electrical focus can sit near zero where a negative eye
        # window cancels the positive craniofacial field. Read the surrounding
        # *prior* voltage as the permissive signal, preserving voltage-first
        # timing and causal sensitivity to a matched voltage clamp.
        face_ready = ((F.avg_pool2d(x[:,4:5].abs(), 5, 1, 2) - .08) * 9).clamp(0, 1)
        ready = (1-face) + face * face_ready
        # Boundary-facing organizers are computed from the current tissue,
        # never from absolute image coordinates or an anatomical target map.
        left_boundary = live * (1 - shift(live, 0, -1))
        right_boundary = live * (1 - shift(live, 0, 1))
        # Organizer choice reads the previous bioelectric field. Spatial
        # averaging lets an eye's local hyperpolarization coexist with a
        # depolarized surrounding head territory.
        organizer_voltage = (F.avg_pool2d(x[:,4:5]*live,5,1,2)
                             / F.avg_pool2d(live,5,1,2).clamp_min(1e-4))
        voltage_head_gate = torch.sigmoid(18*(organizer_voltage-.03))
        voltage_tail_gate = torch.sigmoid(18*(-organizer_voltage-.03))
        # Slow concordant connections are needed to express a *new* organizer
        # after washout. Electrical repatterning precedes this BDH-supported
        # anatomical readout; an ablated loop leaves only a subthreshold trace.
        nearby_slow = F.max_pool2d(es[:,:4].mean(1,keepdim=True),9,1,4)
        normalized_loop_gain = (torch.tanh(self.loop_gain.clamp_min(0))/.46211717).clamp(0,2)
        loop_support = (.12 + .88*normalized_loop_gain*(nearby_slow/.20).clamp(0,1)).clamp(0,1)
        anterior_source = (left_boundary * torch.sigmoid(16 * (polarity - .08))
                           * voltage_head_gate)
        # A jagged new posterior rim can make one organizer appear as two
        # disconnected head islands. Read its voltage/BDH-supported strength
        # into a single centerline field from the surviving structural chart.
        # Read only the posterior organizer territory. Otherwise a nearby
        # unperturbed trunk cell can bypass a posterior voltage clamp when
        # the reduction selects the strongest cell anywhere in the body.
        posterior_readout_region = (axial > .45*major).to(x.dtype)
        posterior_drive = (posterior_program * voltage_head_gate * loop_support
                           * live * posterior_readout_region).amax((-2,-1), keepdim=True)
        posterior_head_field = torch.exp(-.5*((axial-.75*major)/2.2).square()
                                         -.5*(lateral_axis/2.2).square())
        posterior_head = (1.1*posterior_drive*posterior_head_field).clamp(0,1)
        plan_head = ((4.8*F.avg_pool2d(anterior_source, 5, 1, 2)).clamp(0,1)
                     + posterior_head).clamp(0,1)*live
        # The ragged posterior edge can split a boundary-averaged tail into
        # disconnected lobes on an otherwise intact animal. A single smooth
        # centerline field supplies shape; the prior voltage, polarity, and
        # posterior organizer still decide whether that field becomes tail.
        tail_field = torch.exp(-.5*((axial-.75*major)/1.6).square()
                               -.5*(lateral_axis/1.9).square())
        tail_fate = ((1-posterior_program) * torch.sigmoid(16*(-polarity-.08))
                     * voltage_tail_gate
                     * (1 - (1.2 * coupling_signal).clamp(0, 1))
                     * (1 - (1.4 * posterior_drive).clamp(0, 1)))
        plan_tail = (1.25*tail_field*tail_fate).clamp(0,1)*live

        # Shared eye program: each active end recruits two lateral primordia.
        # Coordinates come from surviving tissue's coarse structural prior,
        # avoiding fragmented eyes on a ragged regenerating boundary.
        eye_axial_fraction = .70 + .06 * torch.tanh(self.eye_axial_bias)
        eye_lateral_offset = 1.7 + .4 * torch.tanh(self.eye_lateral_bias)
        eye_axial = torch.exp(-.5*((axial.abs()-eye_axial_fraction*major)/1.0).square())
        eye_lateral = torch.exp(-.5*((lateral_axis.abs()-eye_lateral_offset)/.65).square())
        left_program = torch.sigmoid(-2*axial) * torch.sigmoid(16*(polarity-.08))
        right_program = torch.sigmoid(2*axial) * posterior_drive
        eye_organizer = (left_program*voltage_head_gate+right_program).clamp(0,1)
        plan_eye = (1.3*eye_axial*eye_lateral*eye_organizer + .7*eye_memory).clamp(0,1)*live

        # The same physical-to-voltage-to-fate ordering applies to the face
        # domain. These synthetic templates express a protocol, not a teacher.
        face_head = torch.sigmoid(15 * (voltage - .09)) * live
        # A narrow physical-voltage window sets the eye row; the structural
        # chart sets its lateral pair. The cue is upstream of visible anatomy.
        face_eye_axial = torch.exp(-.5*((physical-.07)/.023).square())
        face_eye_lateral = torch.exp(-.5*((lateral_axis.abs()-.38*minor)/.95).square())
        face_eye = (1.25*face_eye_axial*face_eye_lateral*live).clamp(0,1)
        target_head = (1-face) * plan_head + face * face_head
        target_eye = (1-face) * plan_eye + face * face_eye
        target_tail = (1-face) * plan_tail

        residual_input = torch.cat((new_body, voltage, polarity, head_memory,
                                    eye_memory, activator, inhibitor, regulatory, structure), 1)
        residual = .025 * torch.tanh(self.residual(residual_input))
        head_rate = .075 * self.head_log_gain.exp().clamp(.5, 2)
        eye_rate = .085 * self.eye_log_gain.exp().clamp(.5, 2)
        head = (head + fire * ready * (head_rate * (target_head - head) + residual[:, 0:1])).clamp(0, 1) * live
        eye = (eye + fire * ready * (eye_rate * (target_eye - eye) + residual[:, 1:2])).clamp(0, 1) * live
        tail = (tail + fire * .075 * ((target_tail - tail) + residual[:, 2:3])).clamp(0, 1) * live
        activator = (activator + fire * .045 * (head - .6*eye - activator)).clamp(0, 1) * live
        inhibitor = (inhibitor + fire * .035 * (.25*new_body + .65*eye + .4*tail - inhibitor)).clamp(0, 1) * live
        regulatory = (regulatory + fire * ready * (.05 * (torch.tanh(2*x[:, 4:5])-regulatory)
                                                   + residual[:, 3:4])).clamp(-1, 1) * live
        structure = (structure + fire * ready * (.025 * (.75*new_body-structure)
                                                 + residual[:, 4:5])).clamp(0, 1) * live
        polarity = (polarity + fire * face * .012 * (torch.tanh(3*physical)-polarity)).clamp(-1, 1) * live
        trace = (x[:, 20:21] + fire * .04 * (voltage-x[:, 20:21])).clamp(-1, 1) * live
        preference = (x[:, 21:22] + fire * .003 * (trace*regulatory-x[:, 21:22])).clamp(-1, 1) * live

        # Reciprocal voltage-concordant edges form quickly, then consolidate.
        offsets = ((-1,0),(1,0),(0,-1),(0,1),
                   (0,-self.config.long_radius),(0,self.config.long_radius))
        neigh_v = torch.cat([shift(voltage,*o) for o in offsets],1)
        neigh_live = torch.cat([shift(live,*o) for o in offsets],1)
        open_local = (1-block) * torch.cat([shift(1-block,*o) for o in offsets],1)
        long_open = (1-long_block) * torch.cat([shift(1-long_block,*o) for o in offsets],1)
        conductance = torch.cat((open_local[:,:4],open_local[:,4:]*long_open[:,4:]),1)
        eligible = (1-.5*(neigh_v-voltage).abs()).clamp(0,1) * live * neigh_live * conductance
        neigh_memory = torch.cat([shift(head_memory,*o) for o in offsets],1)
        memory_loop = (.30*torch.tanh(self.loop_gain.clamp_min(0))*head_memory
                       * neigh_memory*live*neigh_live*conductance)
        eligible = (eligible+memory_loop).clamp(0,1)
        ef = (ef + fire * (.035*eligible*(1-ef-es).clamp_min(0)-.015*ef)).clamp(0,1)
        es = (es + fire * (.002*ef-.00035*es)).clamp(0,1)
        valid = live * neigh_live
        ef, es = ef*valid, es*valid
        # One-tick delayed axial packets preserve an inspectable long-range
        # communication state, with no phenotype map encoded in the packet.
        packet = torch.cat((voltage,polarity,regulatory,head_memory,eye_memory,
                            activator,inhibitor,new_body,voltage,live),1)
        delayed = torch.stack((shift(packet,0,-self.config.long_radius),
                               shift(packet,0,self.config.long_radius)),1) * valid[:,4:,None]

        cells = torch.cat((new_body,head,eye,tail,voltage,activator,inhibitor,
                           physical, new_budget, code_next[:,:3], regulatory,polarity,
                           structure,x[:,15:16]*live,code_next[:,3:5],x[:,18:20]*live,
                           trace,preference,x[:,22:23]*live,
                           face*live),1)
        induction = torch.cat((head_memory,eye_memory,injury),1)
        return ProtocolState(cells,ef,es,delayed,induction)
