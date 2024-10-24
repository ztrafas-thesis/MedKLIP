# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
from sklearn.metrics import log_loss
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
from .backbone import build_backbone
from .deformable_transformer import DeformableTransformer
from .util.misc import NestedTensor, nested_tensor_from_tensor_list
import torchvision.models as models
from einops import rearrange
from transformers import AutoModel
'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''




class MedKLIP(nn.Module):

    def __init__(self, config, ana_book, disease_book):
        super(MedKLIP, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(config['text_encoder'],freeze_layers = None).to(disease_book['input_ids'].device)
            self.disease_book = bert_model(input_ids = disease_book['input_ids'],attention_mask = disease_book['attention_mask'])#(**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:,0,:]
        self.disease_embedding_layer = nn.Linear(768,256)
        self.cl_fc = nn.Linear(256,768)
        
        
        ''' visual backbone'''
        if config['deformable'] == False:   
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                                "resnet50": models.resnet50(pretrained=False)}
            resnet = self._get_res_basemodel(config['res_base_model'])
            num_ftrs = int(resnet.fc.in_features/2)
            self.res_features = nn.Sequential(*list(resnet.children())[:-3])
            self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
            self.res_l2 = nn.Linear(num_ftrs, self.d_model)
        else:
            self.backbone = build_backbone(config)
            if config['num_feature_levels'] > 1:
                num_backbone_outs = len(self.backbone.strides)
                input_proj_list = []
                for _ in range(num_backbone_outs):
                    in_channels = self.backbone.num_channels[_]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    ))
                for _ in range(config['num_feature_levels'] - num_backbone_outs):
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    ))
                    in_channels = self.d_model
                self.input_proj = nn.ModuleList(input_proj_list)
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(self.backbone.num_channels[0], self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )])


        ###################################
        ''' Query Decoder'''
        ###################################
        if config['deformable'] == False:
            self.H = config['H'] 
            decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                            0.1, 'relu',normalize_before=True)
            decoder_norm = nn.LayerNorm(self.d_model)
            self.decoder = TransformerDecoder(decoder_layer, config['N'] , decoder_norm,
                                    return_intermediate=False)
        else:
            self.transformer = DeformableTransformer(num_encoder_layers=6, num_decoder_layers=6, num_feature_levels=config['num_feature_levels'],nhead=8)

        # Learnable Queries
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        # self.classifier = nn.Linear(self.d_model,config['attribute_set_size'])

        # Class classifier
        self.cls_classifier = nn.Linear(self.d_model,config['num_classes'])

        # self.apply(self._init_weights)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model
    
    def image_encoder(self, xis):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        batch_size = xis.shape[0]
        res_fea = self.res_features(xis) #batch_size,feature_size,patch_num,patch_num
        res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)
        
        
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        return out_emb

    def forward(self, images, return_intermediate=False):
        B = images.shape[0]
        
        device = images.device
        ''' Visual Backbone '''
        if self.config['deformable'] == False:
            x = self.image_encoder(images) #batch_size,patch_num,dim
            features = x.transpose(0,1) #patch_num b dim
            # print(features.shape)
            # torch.Size([196, 2, 256])
        
        else:
            samples = nested_tensor_from_tensor_list(images) # NestedTensor(images, torch.ones(B, images.size(2), images.size(3))).to(device)
            features, pos = self.backbone(samples)

            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            if self.config['num_feature_levels'] > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.config['num_feature_levels']):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)

            # print('srcs:', [src.shape for src in srcs])
            # print('masks:', [mask.shape for mask in masks])
            # srcs: [torch.Size([2, 256, 7, 7]), torch.Size([2, 256, 4, 4]), torch.Size([2, 256, 2, 2]), torch.Size([2, 256, 1, 1])]
            # masks: [torch.Size([2, 7, 7]), torch.Size([2, 4, 4]), torch.Size([2, 2, 2]), torch.Size([2, 1, 1])]

            # features = srcs[1].flatten(2)  # [2, 256, 49]
            # features = features.permute(2, 0, 1)
        
        
        query_embed = self.disease_embedding_layer(self.disease_book)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)

        if self.config['deformable'] == False:
            features,ws = self.decoder(query_embed, features, 
                memory_key_padding_mask=None, pos=None, query_pos=None)
            
        else:
            # print('query_embed:',query_embed.shape)
            # query_embed: torch.Size([75, 2, 256])
            features = self.transformer(srcs, masks, pos, query_embed)

        if return_intermediate:
            return features
        out = self.dropout_feas(features)
        x= self.cls_classifier(out).transpose(0,1) #B query Atributes

        x = x.mean(dim=1) 
        
        return x
        



    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()