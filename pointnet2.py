import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation
from .hyptorch.nn import MobiusLayer, ToPoincare


class SharedHypProjector(nn.Module):
    def __init__(self, cfg):
        super(SharedHypProjector, self).__init__()
        self.hyp_c = getattr(cfg.TRAIN, 'HYP_C', 0.1)
        self.tangent_scale = float(getattr(cfg.TRAIN, 'HYP_TANGENT_SCALE', 1.0))
        if self.tangent_scale <= 0:
            raise ValueError('HYP_TANGENT_SCALE 必须大于 0。')
        self.pre_poincare = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        self.to_poincare = ToPoincare(c=self.hyp_c, clip_r=getattr(cfg.TRAIN, 'HYP_CLIP_R', 2.3))

    def forward(self, feat):
        shared_feat = self.pre_poincare(feat)
        tangent_feat = shared_feat * self.tangent_scale
        hyp_feat = self.to_poincare(tangent_feat.permute(0, 2, 1).contiguous())
        return shared_feat, hyp_feat.permute(0, 2, 1).contiguous()


class PointNet2(nn.Module):
    def __init__(self, cfg=None, args=None):
        super(PointNet2, self).__init__()
        self.cfg = cfg
        self.num_class = cfg.DATASET.DATA.LABEL_NUMBER
        self.level = args.mc_level
        self.num_levels = len(self.num_class)

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        if self.level != -1 and (self.level < 0 or self.level >= self.num_levels):
            raise ValueError('mc_level must be -1 or in [0, {}], got {}'.format(self.num_levels - 1, self.level))

        self.decoders = nn.ModuleList([
            PointNet2Decoder(num_class, self.cfg)
            for num_class in self.num_class
        ])

        # The baseline (euclidean_raw) does not use the hyperbolic projector,
        # so it is omitted to keep parameter counts faithful to the ablation.
        self.seg_head_type = getattr(cfg.TRAIN, 'SEG_HEAD_TYPE', 'mobius').lower()
        if self.seg_head_type != 'euclidean_raw':
            self.shared_hyp_projector = SharedHypProjector(self.cfg)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        input_list = [l0_xyz, l0_points,
                      l1_xyz, l1_points,
                      l2_xyz, l2_points,
                      l3_xyz, l3_points,
                      l4_xyz, l4_points]

        if self.level == -1:
            outputs = []
            hyp_feats = []
            for decoder in self.decoders:
                x, feat = self._forward_level(decoder, input_list)
                outputs.append(x)
                hyp_feats.append(feat)
            return outputs, hyp_feats

        return self._forward_level(self.decoders[self.level], input_list)

    def _forward_level(self, decoder, input_list):
        feat = decoder(*input_list, return_feat_only=True)
        if decoder.seg_head_type == 'euclidean_raw':
            x = decoder.classify(feat=feat)
            return x, None
        shared_feat, hyp_feat = self.shared_hyp_projector(feat)
        x = decoder.classify(feat=shared_feat, hyp_feat=hyp_feat)
        return x, hyp_feat


class PointNet2Decoder(nn.Module):
    def __init__(self, num_class, cfg):
        super(PointNet2Decoder, self).__init__()
        self.cfg = cfg
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(cfg.TRAIN.DROPOUT_RATE)
        self.seg_head_type = getattr(cfg.TRAIN, 'SEG_HEAD_TYPE', 'mobius').lower()
        self.hyp_c = getattr(cfg.TRAIN, 'HYP_C', 0.1)
        self.tangent_scale = float(getattr(cfg.TRAIN, 'HYP_TANGENT_SCALE', 1.0))
        if self.tangent_scale <= 0:
            raise ValueError('HYP_TANGENT_SCALE 必须大于 0。')
        self.to_poincare = ToPoincare(c=self.hyp_c, clip_r=getattr(cfg.TRAIN, 'HYP_CLIP_R', 2.3))
        if self.seg_head_type == 'mobius':
            self.mobius_out = MobiusLayer(128, num_class, c=self.hyp_c)
            self.logit_scale = float(getattr(cfg.TRAIN, 'MOBIUS_LOGIT_SCALE', 10.0))
            if self.logit_scale <= 0:
                raise ValueError('MOBIUS_LOGIT_SCALE 必须大于 0；若不希望额外放大，请设为 1.0。')
        elif self.seg_head_type in ('euclidean_hyp', 'euclidean', 'euclidean_raw'):
            self.conv_out = nn.Conv1d(128, num_class, 1)
            self.logit_scale = 1.0
        else:
            raise ValueError('Unknown SEG_HEAD_TYPE: {}'.format(self.seg_head_type))

    def encode(self, l0_xyz, l0_points,
               l1_xyz, l1_points,
               l2_xyz, l2_points,
               l3_xyz, l3_points,
               l4_xyz, l4_points):
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        return feat

    def classify(self, feat=None, hyp_feat=None):
        if self.seg_head_type == 'mobius':
            if hyp_feat is None:
                raise ValueError('Mobius head requires hyp_feat.')
            x = self.mobius_out(hyp_feat.permute(0, 2, 1).contiguous()) * self.logit_scale
            return x.permute(0, 2, 1).contiguous()
        if self.seg_head_type == 'euclidean_hyp':
            if hyp_feat is None:
                raise ValueError('euclidean_hyp head requires hyp_feat.')
            return self.conv_out(hyp_feat)
        if feat is None:
            raise ValueError('Euclidean head requires feat.')
        x = self.drop1(feat)
        x = self.conv_out(x)
        return x

    def forward(self, l0_xyz, l0_points,
                l1_xyz, l1_points,
                l2_xyz, l2_points,
                l3_xyz, l3_points,
                l4_xyz, l4_points,
                return_feat_only=False,
                feat=None,
                hyp_feat=None):
        if feat is None:
            feat = self.encode(
                l0_xyz, l0_points,
                l1_xyz, l1_points,
                l2_xyz, l2_points,
                l3_xyz, l3_points,
                l4_xyz, l4_points,
            )
        if return_feat_only:
            return feat
        if self.seg_head_type == 'mobius' and hyp_feat is None:
            tangent_feat = feat * self.tangent_scale
            hyp_feat = self.to_poincare(tangent_feat.permute(0, 2, 1).contiguous())
            hyp_feat = hyp_feat.permute(0, 2, 1).contiguous()
        x = self.classify(feat=feat, hyp_feat=hyp_feat)
        return x, hyp_feat


if __name__ == '__main__':
    device = torch.device("cuda")
    model = PointNet2().to(device)
    rand_input = torch.rand(32, 6, 2048).to(device)
    output = model(rand_input)
    print(output)
