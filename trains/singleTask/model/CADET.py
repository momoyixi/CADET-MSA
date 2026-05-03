"""
here is the mian backbone for CADET
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder   

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class CADET(nn.Module):
    def __init__(self, args):
        super(CADET, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads     
        self.layers = args.nlevels 

        self.reverse_grad_weight = args.reverse_grad_weight
        self.use_cmd_sim = args.use_cmd_sim

        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a    
        combined_dim_high = self.d_a 
        combined_dim = (self.d_l + self.d_a + self.d_v ) + self.d_l * 3  
        
        output_dim = 1

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        self.encoder_s_l = self.get_network(self_type='l', layers = self.layers)       
        self.encoder_s_v = self.get_network(self_type='v', layers = self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers = self.layers)

        self.encoder_c = self.get_network(self_type='l', layers = self.layers)        
        
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)     
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)


        self.decoder_A_from_L = nn.Conv1d(self.d_l * 2, self.d_a, kernel_size=1, padding=0, bias=False)
        self.decoder_L_from_A = nn.Conv1d(self.d_a * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_V_from_L = nn.Conv1d(self.d_l * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_L_from_V = nn.Conv1d(self.d_v * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        
        self.discriminator_A = nn.Sequential(
            nn.Linear(self.d_a, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.discriminator_V = nn.Sequential(
            nn.Linear(self.d_v, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.d_l, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.sp_discriminator = nn.Sequential(
            nn.Linear(self.d_l, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)     
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        self.trans_l_with_v = TextEnhancedTransformer(
            embed_dim=self.d_a,
            num_heads=self.num_heads,
            layers=2,
            attn_dropout=self.attn_dropout_a,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout
        )
        
        self.trans_l_with_a = TextEnhancedTransformer(
            embed_dim=self.d_v,
            num_heads=self.num_heads,
            layers=2,
            attn_dropout=self.attn_dropout_v,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout
        )
        self.trans_v_with_a = self.get_network(self_type='va', layers = self.layers)  
        self.trans_a_with_v = self.get_network(self_type='av', layers = self.layers) 
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)
        
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v        
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training) 
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l) 
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a) 
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)   
        proj_x_v = proj_x_v .permute(2, 0, 1)  
        proj_x_a = proj_x_a.permute(2, 0, 1)

        s_l = self.encoder_s_l(proj_x_l)    
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)


        s_l = s_l.permute(1, 2, 0)   
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]

        if not self.use_cmd_sim:
            reversed_c_l = ReverseLayerF.apply(c_l, self.reverse_grad_weight)
            reversed_c_v = ReverseLayerF.apply(c_v, self.reverse_grad_weight)
            reversed_c_a = ReverseLayerF.apply(c_a, self.reverse_grad_weight)
            
            domain_input_l = reversed_c_l.mean(dim=-1)
            domain_input_v = reversed_c_v.mean(dim=-1)
            domain_input_a = reversed_c_a.mean(dim=-1)
            
            self.domain_label_l = self.domain_discriminator(domain_input_l)
            self.domain_label_v = self.domain_discriminator(domain_input_v)
            self.domain_label_a = self.domain_discriminator(domain_input_a)
        else:
            self.domain_label_l = None
            self.domain_label_v = None
            self.domain_label_a = None

        private_input_l = s_l.mean(dim=-1)  
        private_input_v = s_v.mean(dim=-1)
        private_input_a = s_a.mean(dim=-1)
        shared_input = (c_l + c_v + c_a).mean(dim=-1) / 3.0
        special_input = (s_l + s_v + s_a).mean(dim=-1) / 3.0
        self.shared_or_private_l = self.sp_discriminator(private_input_l)
        self.shared_or_private_v = self.sp_discriminator(private_input_v)
        self.shared_or_private_a = self.sp_discriminator(private_input_a)
        self.shared_or_private_c = self.sp_discriminator(shared_input)
        self.shared_or_private_s = self.sp_discriminator(special_input)

        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))
        
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        gen_A = self.decoder_A_from_L(torch.cat([s_l, c_l], dim=1))
        gen_A_permuted = gen_A.permute(2, 0, 1)  # (seq_len, batch, d_a)
        
        s_a_prime = self.encoder_s_a(gen_A_permuted).permute(1, 2, 0)  # (batch, d_a, seq_len)
        c_a_prime = self.encoder_c(gen_A_permuted).permute(1, 2, 0)
        
        recon_L_prime = self.decoder_L_from_A(torch.cat([s_a_prime, c_a_prime], dim=1))
        
        real_A = proj_x_a.permute(1, 2, 0).mean(dim=-1)  # (batch, d_a)
        fake_A = gen_A.mean(dim=-1)
        d_real_A = self.discriminator_A(real_A)
        d_fake_A = self.discriminator_A(fake_A)

        gen_V = self.decoder_V_from_L(torch.cat([s_l, c_l], dim=1))
        gen_V_permuted = gen_V.permute(2, 0, 1)  # (seq_len, batch, d_v)

        s_v_prime = self.encoder_s_v(gen_V_permuted).permute(1, 2, 0)  # (batch, d_v, seq_len)
        c_v_prime = self.encoder_c(gen_V_permuted).permute(1, 2, 0)

        recon_L_prime_v = self.decoder_L_from_V(torch.cat([s_v_prime, c_v_prime], dim=1))

        real_V = proj_x_v.permute(1, 2, 0).mean(dim=-1)  # (batch, d_v)
        fake_V = gen_V.mean(dim=-1)
        d_real_V = self.discriminator_V(real_V)
        d_fake_V = self.discriminator_V(fake_V)



        recon_l = recon_l.permute(2, 0, 1)  
        recon_v = recon_v.permute(2, 0, 1)   
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l).permute(1, 2, 0)                                                                                             
        s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)
        
        s_l = s_l.permute(2, 0, 1)  
        s_v = s_v.permute(2, 0, 1)   
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)
       
       #enhancement
        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)  
        repr_l_low = self.proj1_l_low(hs_l_low)                            
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low         
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        
        c_l_att = self.self_attentions_c_l(c_l)  
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)   

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion                       
        logits_c = self.out_layer_c(c_proj)     
                      
        h_ls = s_l                     
        h_ls = self.trans_l_mem(h_ls)  
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  

        h_l_with_as = self.trans_l_with_a(s_a,s_a,s_l)
        h_as = h_l_with_as
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]  

        h_l_with_vs = self.trans_l_with_v(s_v,s_v,s_l)  
        h_vs = h_l_with_vs
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]  


        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)
        
        #fusion
        last_h_l = torch.sigmoid(self.projector_l(hs_proj_l_high))   
        last_h_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        last_h_a = torch.sigmoid(self.projector_a(hs_proj_a_high))
        c_fusion = torch.sigmoid(self.projector_c(c_fusion))
        
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)   

        #prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs                          

        output = self.out_layer(last_hs_proj)

        res = {
            'origin_l': proj_x_l.permute(1, 2, 0),
            'origin_v': proj_x_v.permute(1, 2, 0),
            'origin_a': proj_x_a.permute(1, 2, 0),
            's_l': s_l,        
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l.permute(1, 2, 0),
            'recon_v': recon_v.permute(1, 2, 0),
            'recon_a': recon_a.permute(1, 2, 0),
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,           
            'logits_v_hetero': logits_v_high, 
            'logits_a_hetero': logits_a_high,
            'logits_c': logits_c,
            'output_logit': output,
            'last_hs_proj': last_hs_proj,  
            'gen_A': gen_A,
            'gen_V': gen_V,
            'recon_L_prime': recon_L_prime,
            'recon_L_prime_v': recon_L_prime_v,
            'd_real_A': d_real_A,
            'd_fake_A': d_fake_A,
            'd_real_V': d_real_V,
            'd_fake_V': d_fake_V,
            'domain_label_l': self.domain_label_l,
            'domain_label_v': self.domain_label_v,
            'domain_label_a': self.domain_label_a,
            'shared_or_private_l': self.shared_or_private_l,
            'shared_or_private_v': self.shared_or_private_v,
            'shared_or_private_a': self.shared_or_private_a,
            'shared_or_private_c': self.shared_or_private_c,

        }
        return res

        
class TextEnhancedTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, embed_dropout) -> None:
        super().__init__()
        self.lower_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=1,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            attn_mask=True
        )
        self.upper_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            attn_mask=True
        )
    
    def forward(self, query_m, key_m, text):
        c = self.lower_mha(text)          
        enhanced = self.upper_mha(c,query_m,query_m)  
        return enhanced


