import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .operation import FactorizedReduce, ReLUConvBN, OPS
from .genotype import  s3
from nas.Mix import mixnet_xl
from .RayNet import ASSP, SepConv
class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction_prev):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)  # 8
        self._steps = len(op_names) // 2  # 4
        self._concat = concat
        self.multiplier = len(concat)  # 4

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class CellDecode(nn.Module):
    "Cell structure for searching"

    def __init__(self, genotype, c_pp, c_p, C,  expansion=True, expansion_prev=False):
        super(CellDecode, self).__init__()
        self.expansion = expansion
        self.expansion_prev = expansion_prev
        self.preprocess0 = ReLUConvBN(c_pp, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(c_p, C, 1, 1, 0, affine=False)
        op_names, indices = zip(*genotype.reduce)
        concat = genotype.reduce_concat
        self._compile(C, op_names, indices, concat, expansion)

    def _compile(self, C, op_names, indices, concat, expansion):
        assert len(op_names) == len(indices)  # 8
        self._steps = len(op_names) // 2  # 4
        self._concat = concat
        self.multiplier = len(concat)  # 4

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride =1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices


    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        if self.expansion_prev:
            s0 = F.interpolate(s0, scale_factor=2.,
                               mode='bilinear', align_corners=True)
        s1 = self.preprocess1(s1)
        if self.expansion:
            s0 = F.interpolate(s0, scale_factor=2.,
                               mode='bilinear', align_corners=True)
            s1 = F.interpolate(s1, scale_factor=2.,
                               mode='bilinear', align_corners=True)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

class NASseg(nn.Module):
    "Search unet like network "

    def __init__(self, C, num_classes, genotype='unet', layers=12):
        super(NASseg, self).__init__()
        self.channel = C

        self.num_classes = num_classes
        self.layer = layers
        self.stem0 = nn.Sequential(nn.Conv2d(3, C//2, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(C//2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(C//2, C, 3, stride=2,
                                             padding=1, bias=False),
                                   nn.BatchNorm2d(C))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(
            C, C, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(C))

        c_pp, c_p, c_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//4, 2*layers//4, 3*layers//4]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, c_pp, c_p, c_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            c_pp, c_p = c_p, cell.multiplier*c_curr

        self.cells_decode = nn.ModuleList()
        reduction = False
        expansion_prev = False
        for i in range(layers):
            if i+1 in [layers//4, 2*layers//4, 3*layers//4]:
                expansion = True
                c_curr //= 2
            else:
                expansion = False

            cell = CellDecode(genotype, c_pp, c_p, c_curr,
                              expansion=expansion, expansion_prev=expansion_prev)
            expansion_prev = expansion
            self.cells_decode += [cell]
            c_pp, c_p = c_p, c_curr * cell.multiplier

        self.outputlayer = nn.Conv2d(64, num_classes, 1, 1)
        self._init_params()

    def forward(self, inputs):
        N, C, W, H = inputs.size()
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        middle_feature = []
        for _, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            middle_feature.append(s1)
        for i, cell in enumerate(self.cells_decode):
            s0, s1 = s1, cell(s0, s1 )
            if cell.expansion:
                # feature=F.interpolate(middle_feature[self.layer-i-1],scale_factor=2.,mode='bilinear',align_corners=True)
                feature = middle_feature[self.layer-i-1]
                s0 = s0+feature
        output = self.outputlayer(s1)
        output = F.interpolate(
            output, (W, H), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        return output

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)



class NASRayNetEval(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, genotype='ray1',layer=12):
        super(NASRayNetEval, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained,
                                num_classes=num_classes,head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.aspp = ASSP(in_channels=320, output_stride=16)
        self.decode_cell = CellDecode(genotype, 256, 128, 64, expansion_prev=True)

        self.low_cell1 = Cell(genotype, 48, 64, 16,reduction=False, reduction_prev=True)
        self.low_cell2 = Cell(genotype, 64,  64, 16,reduction=False, reduction_prev=False)
        self.low_cell3 = Cell(genotype, 64, 64, 16,reduction=False, reduction_prev=False)
        self.low_cell4 = Cell(genotype, 64, 64, 16,reduction=False, reduction_prev=False)

        self.outcell1 = CellDecode(genotype,256, 64, 32,expansion=True, expansion_prev=True)
        self.cell1 = Cell(genotype, 48, 128, 32,reduction=False, reduction_prev=False)
        self.cell2 = Cell(genotype, 128, 128, 32,reduction=False, reduction_prev=False)
        self.cell3 = Cell(genotype, 128, 128, 32,reduction=False, reduction_prev=False)

        self.out = SepConv(128, num_classes, 1, 1, 0)
        self.up4 = nn.Upsample( scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-1])

        decode1 = self.decode_cell(aspp, middle_feature[-2])


        low_feat1 = self.low_cell1( middle_feature[0], middle_feature[1])
        low_feat2=self.low_cell2(middle_feature[1],low_feat1)
        low_feat3=self.low_cell3(low_feat1,low_feat2)
        low_feat4=self.low_cell4(low_feat2,low_feat3)
        out1 = self.outcell1(decode1, low_feat4)
        out2 = self.cell1(middle_feature[0],out1)
        out3= self.cell2(out1,out2)
        out4= self.cell3(out2,out3)
        out = self.out(out4)
        out = self.up4(out)
        out = torch.softmax(out, 1)
        return out


class NASRayNetEvalDense(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, genotype='ray1',layer=12):
        super(NASRayNetEvalDense, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained,
                                num_classes=num_classes,head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.aspp = ASSP(in_channels=320, output_stride=16)
        self.decode_cell = CellDecode(genotype, 256, 192, 64, expansion_prev=True)

        self.low_cell1 = Cell(genotype, 48, 64, 16,reduction=False, reduction_prev=True)
        self.low_cell2 = Cell(genotype, 64,  128, 16,reduction=False, reduction_prev=False)
        self.low_cell3 = Cell(genotype, 128, 192, 16,reduction=False, reduction_prev=False)
        self.low_cell4 = Cell(genotype, 192, 256, 16,reduction=False, reduction_prev=False)

        self.outcell1 = CellDecode( genotype,256, 64, 16,expansion=True, expansion_prev=True)
        self.cell1 = Cell(genotype, 48, 112, 16,reduction=False, reduction_prev=False)
        self.cell2 = Cell(genotype, 112, 176, 16,reduction=False, reduction_prev=False)
        self.cell3 = Cell(genotype, 176, 240, 16,reduction=False, reduction_prev=False)
        self.cell4 = Cell(genotype, 240, 304, 16,reduction=False, reduction_prev=False)
        self.cell5 = Cell(genotype, 304, 368, 16,reduction=False, reduction_prev=False)

        self.out = SepConv(432, num_classes, 1, 1, 0)
        self.up4 = nn.Upsample( scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-1])

        decode1 = self.decode_cell(aspp, middle_feature[-2])


        low_feat1 = self.low_cell1( middle_feature[0], middle_feature[1])
        low_feat1=torch.cat([middle_feature[1],low_feat1],1)
        low_feat2=self.low_cell2(middle_feature[1],low_feat1)
        low_feat2=torch.cat([low_feat1,low_feat2],1)

        low_feat3=self.low_cell3(low_feat1,low_feat2)
        low_feat3=torch.cat([low_feat2,low_feat3],1)
        low_feat4=self.low_cell4(low_feat2,low_feat3)


        out1 = self.outcell1(decode1, low_feat4)
        out1=torch.cat([middle_feature[0],out1],1)
        out2 = self.cell1(middle_feature[0],out1)
        out2=torch.cat([out1,out2],1)
        out3= self.cell2(out1,out2)
        out3=torch.cat([out2,out3],1)
        out4= self.cell3(out2,out3)
        out4=torch.cat([out3,out4],1)

        out5=self.cell4(out3,out4)

        out5=torch.cat([out4,out5],1)
        out6=self.cell5(out4,out5)
        out6=torch.cat([out5,out6],1)
        out = self.out(out6)
        out = self.up4(out)
        out = torch.softmax(out, 1)
        return out

class NASRayNetEvalDense_v1(nn.Module):
    "adopt from raynet_v0"
    def __init__(self, pretrained=True, num_classes=3, genotype='ray1',layer=12):
        super(NASRayNetEvalDense_v1, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained, num_classes=num_classes,head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.decode_cell = CellDecode(genotype, 128, 192, 64, expansion_prev=False)
        self.low_cell1 = Cell(genotype, 48, 64, 16,reduction=False, reduction_prev=True)
        self.low_cell2 = Cell(genotype, 64,  112, 16,reduction=False, reduction_prev=False)
        self.low_cell3 = Cell(genotype, 112, 128, 16,reduction=False, reduction_prev=False)
        self.low_cell4 = Cell(genotype, 128, 192, 16,reduction=False, reduction_prev=False)

        self.outcell1 = CellDecode( genotype,256, 64, 32,expansion=False, expansion_prev=True)
        self.cell1 = Cell(genotype, 64, 192, 32,reduction=False, reduction_prev=False)
        self.cell2 = Cell(genotype, 192, 192, 16,reduction=False, reduction_prev=False)
        self.cell3 = Cell(genotype, 192, 256, 16,reduction=False, reduction_prev=False)
        self.cell4 = Cell(genotype, 256, 320, 16,reduction=False, reduction_prev=False)
        self.cell5 = Cell(genotype, 320, 384, 16,reduction=False, reduction_prev=False)

        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.out = SepConv(448, num_classes, 1, 1, 0)
        self.up4 = nn.Upsample( scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)

        decode1 = self.decode_cell(middle_feature[-4], middle_feature[-3])


        low_feat1 = self.low_cell1( middle_feature[0], middle_feature[1])

        pool_feature=self.maxpool(middle_feature[0])

        low_feat1=torch.cat([pool_feature,low_feat1],1)

        low_feat2=self.low_cell2(middle_feature[1],low_feat1)

        low_feat2=torch.cat([middle_feature[1],low_feat2],1)

        low_feat3=self.low_cell3(low_feat1,low_feat2)

        up_feature=F.interpolate(middle_feature[2],scale_factor=2,mode='bilinear',align_corners=True)
        low_feat3=torch.cat([up_feature,low_feat3],1)
        low_feat4=self.low_cell4(low_feat2,low_feat3)


        out1 = self.outcell1(decode1, low_feat4)
        out1=torch.cat([middle_feature[1],out1],1)
        out2 = self.cell1(middle_feature[1],out1)
        out2=torch.cat([middle_feature[1],out2],1)
        out3= self.cell2(out1,out2)
        out3=torch.cat([out2,out3],1)
        out4= self.cell3(out2,out3)
        out4=torch.cat([out3,out4],1)
        out5=self.cell4(out3,out4)

        out5=torch.cat([out4,out5],1)
        out6=self.cell5(out4,out5)
        out6=torch.cat([out5,out6],1)
        out = self.out(out6)
        out = self.up4(out)
        out = torch.softmax(out, 1)
        return out




class NASRayNetEval_v0(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, genotype='ray1',layer=12):
        super(NASRayNetEval_v0, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained, num_classes=num_classes,head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.aspp = ASSP(in_channels=192, output_stride=16)
        self.decode_cell1 = CellDecode(genotype, 192, 256, 64, expansion_prev=False)
        self.decode_cell2 = CellDecode(genotype, 128, 256, 64, expansion_prev=True)

        self.low_cell = Cell(genotype, 48, 64, 16, reduction_prev=True)


        self.outcell1 = CellDecode(genotype,64, 256, 32,expansion_prev=True)
        self.out = SepConv(128, num_classes, 1, 1, 0)
        self.up2 = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-2])

        decode1 = self.decode_cell1( middle_feature[-2],aspp)
        decode2 = self.decode_cell2( middle_feature[-3],decode1)


        low_feat1 = self.low_cell( middle_feature[0], middle_feature[1])

        out1 = self.outcell1(low_feat1,decode2)
        out = self.out(out1)
        out = self.up2(out)
        out = torch.softmax(out, 1)
        return out


class NASRayNetEval_v0_dense(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, genotype='ray1',layer=12):
        super(NASRayNetEval_v0_dense, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained, num_classes=num_classes,head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.aspp = ASSP(in_channels=192, output_stride=16)
        self.decode_cell1 = CellDecode(genotype, 192, 256, 64, expansion_prev=False)
        self.decode_cell2 = CellDecode(genotype, 128, 256, 64, expansion_prev=True)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.low_cell1 = Cell(genotype, 48, 64, 16, reduction_prev=True)
        self.low_cell2 = Cell(genotype, 64, 112, 16, reduction_prev=False)
        self.low_cell3 = Cell(genotype, 112, 128, 16, reduction_prev=False)
        self.low_cell4 = Cell(genotype, 128, 192, 16, reduction_prev=False)


        self.outcell1 = CellDecode(genotype,112, 256, 32,expansion_prev=True)
        self.out = SepConv(128, num_classes, 1, 1, 0)
        self.up2 = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-2])
        decode1 = self.decode_cell1( middle_feature[-2],aspp)
        decode2 = self.decode_cell2( middle_feature[-3],decode1)
        pool_feature=self.maxpool(middle_feature[0])        #96-48                
        low_feat1 = self.low_cell1( middle_feature[0], middle_feature[1])  #48
        low_feat1=torch.cat([pool_feature,low_feat1],1)     
        low_feat2=self.low_cell2(middle_feature[1],low_feat1)
        low_feat2=torch.cat([middle_feature[1],low_feat2],1)
        low_feat3=self.low_cell3(low_feat1,low_feat2)
        up_feature=F.interpolate(middle_feature[2],scale_factor=2,mode='bilinear',align_corners=True)
        low_feat3=torch.cat([up_feature,low_feat3],1)
        low_feat4=self.low_cell4(low_feat2,low_feat3)

        out1 = self.outcell1(low_feat1,decode2)
        out = self.out(out1)
        out = self.up2(out)
        out = torch.softmax(out, 1)
        return out


class NASRayNetEval_v1_dense(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, genotype='ray1',layer=12):
        super(NASRayNetEval_v1_dense, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained, num_classes=num_classes,head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.aspp = ASSP(in_channels=192, output_stride=16)
        self.decode_cell1 = CellDecode(genotype, 256, 256, 64, expansion_prev=False)
        self.decode_cell2 = CellDecode(genotype, 256, 256, 64, expansion_prev=True)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.low_cell1 = Cell(genotype, 48, 64, 16, reduction_prev=True)
        self.low_cell2 = Cell(genotype, 64, 112, 16, reduction_prev=False)
        self.low_cell3 = Cell(genotype, 112, 128, 16, reduction_prev=False)
        self.low_cell4 = Cell(genotype, 128, 192, 16, reduction_prev=False)


        self.outcell1 = CellDecode(genotype,112, 256, 32,expansion_prev=True)
        self.outcell2 = Cell(genotype,48,256,32,reduction_prev=False)
        self.outcell3 = Cell(genotype,48,256,32,reduction_prev=False)
        self.outcell4 = Cell(genotype,48,256,32,reduction_prev=False)
        self.out = SepConv(128, num_classes, 1, 1, 0)
        self.up2 = nn.Upsample( scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-2])
        s0=aspp
        s1=aspp

        s0,s1 =s1, self.decode_cell1( s0,s1)
        s1=torch.cat([middle_feature[-3],s1],1)

        s0,s1 = s1,self.decode_cell2( s0,s1)

        low_s0,low_s1=middle_feature[1],middle_feature[1]
        low_s0,low_s1 =low_s1,self.low_cell1( low_s0,low_s1)  #48

        low_s1=torch.cat([s0,s1],1)
        low_s0,low_s1=self.low_cell2(s0,s1)

        low_s1=torch.cat([s0,s1],1)
        low_s0,low_s1=self.low_cell3(low_s0,low_s1)

        low_s1=torch.cat([s0,s1],1)
        low_s1=self.low_cell4(low_s0,low_s1)

        out1 = self.outcell1(low_s1,s1)

        out1=out1.cat([middle_feature[1],out1],1)
        out1 = self.outcell2(middle_feature[-1])

        out = self.out(out1)

        out = self.up2(out)
        out = torch.softmax(out, 1)
        return out


class NASRayNet_seg(nn.Module):
    def __init__(self,pretrained=True,num_classes=3,genotype="s1"):
        super(NASRayNet_seg,self).__init__()
        self.encode=mixnet_xl(pretrained=pretrained,num_classes=num_classes,head_conv=None)
        self.aspp=ASSP(in_channels=192,output_stride=16)
        self.decode_cell1=CellDecode(genotype,192,256,64)
        self.decode_cell2=CellDecode(genotype,192,256,64,expansion_prev=True)

        self.low_cell1=Cell(genotype,48,64,16,True)
        self.low_cell2=Cell(genotype,64,64,16,False)
        self.outcell1=CellDecode(genotype, 64,256,16,expansion_prev=True)
        self.outcell2=Cell(genotype,64,64,16,reduction_prev=False)
        self.out=SepConv(64,num_classes,1,1,0)
        self.up2=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        
    def forward(self,inputs):
        _,middle_feature=self.encode.forward_features(inputs)
        aspp=self.aspp(middle_feature[-2])
        decode1=self.decode_cell1(middle_feature[-2],aspp)
        decode2=self.decode_cell2(middle_feature[-2],decode1)

        low_feat1=self.low_cell1(middle_feature[0],middle_feature[1])
        low_feat2=self.low_cell2(middle_feature[1],low_feat1)

        out=self.outcell1(low_feat2,decode2)
        out=self.outcell2(out,out)

        out=self.out(out)
        out=self.up2(out)
        out=torch.softmax(out,1)
        return out


class NASRayNet_seg(nn.Module):
    def __init__(self,pretrained=True,num_classes=3,genotype="s1"):
        super(NASRayNet_seg,self).__init__()
        self.encode=mixnet_xl(pretrained=pretrained,num_classes=num_classes,head_conv=None)
        self.aspp=ASSP(in_channels=192,output_stride=16)
        self.decode_cell1=CellDecode(genotype,192,256,64)
        self.decode_cell2=CellDecode(genotype,192,256,64,expansion_prev=True)

        self.low_cell1=Cell(genotype,48,64,16,True)
        self.low_cell2=Cell(genotype,64,64,16,False)

        self.outcell1=CellDecode(genotype, 64,256,16,expansion_prev=True)
        self.outcell2=Cell(genotype,64,64,16,reduction_prev=False)
        self.outcell3=Cell(genotype,128,64,16,reduction_prevFalse)
        self.out=SepConv(64,num_classes,1,1,0)
        self.up2=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.maxpool=nn.MaxPool2d(2,2)
    def forward(self,inputs):
        _,middle_feature=self.encode.forward_features(inputs)
        aspp=self.aspp(middle_feature[-2])
        decode1=self.decode_cell1(middle_feature[-2],aspp)
        decode2=self.decode_cell2(middle_feature[-2],decode1)

        low_feat1=self.low_cell1(middle_feature[0],middle_feature[1])
        down_feature=self.maxpool(middle_feature[0])
        up_feature=self.up2(middle_feature[3])
        low_feat1=torch.cat([low_feat1,middle_feature[1],down_feature,up_feature],1)
        low_feat2=self.low_cell2(middle_feature[1],low_feat1)
        low_feat2=torch.cat([middle_feature[1],low_feat2],1)

        out=self.outcell1(low_feat2,decode2)
        out=self.outcell2(out,out)

        out=self.out(out)
        out=self.up2(out)
        out=torch.softmax(out,1)
        return out



if __name__ == "__main__":
    a = NASRayNetEval_v1_dense(True,3,s3,12)
    inputs=torch.randn(2,3,384,384)
    out=a(inputs)
    print(out.size())
