import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from .transformer import *
from transformers import AutoModel
from einops import rearrange

class ModelRes_ft(nn.Module):
    def __init__(self, config, disease_book, res_base_model,out_size,imagenet_pretrain=False,linear_probe=False):
        super(ModelRes_ft, self).__init__()

        self.d_model = config['d_model']
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(config['text_encoder'],freeze_layers = None).to(disease_book['input_ids'].device)
            self.disease_book = bert_model(input_ids = disease_book['input_ids'],attention_mask = disease_book['attention_mask'])#(**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:,0,:]
        self.disease_embedding_layer = nn.Linear(768,256)
        self.cl_fc = nn.Linear(256,768)

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=imagenet_pretrain),
                            "resnet50": models.resnet50(pretrained=imagenet_pretrain)}
        resnet = self._get_res_basemodel(res_base_model)
        num_ftrs = int(resnet.fc.in_features/2)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, self.d_model)

        self.H = config['H'] 
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, config['N'] , decoder_norm,
                                  return_intermediate=False)

        # Learnable Queries
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        # self.classifier = nn.Linear(self.d_model,config['attribute_set_size'])

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
    
    def forward(self, images,linear_probe=False):
        B = images.shape[0]
        
        device = images.device
        ''' Visual Backbone '''
        x = self.image_encoder(images) #batch_size,patch_num,dim
        features = x.transpose(0,1) #patch_num b dim
        
        query_embed = self.disease_embedding_layer(self.disease_book)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
        features,ws = self.decoder(query_embed, features, 
            memory_key_padding_mask=None, pos=None, query_pos=None)
        out = self.dropout_feas(features)
        x= self.cls_classifier(out).transpose(0,1) #B query Atributes
        
        x = x.mean(dim=1) 

        return x