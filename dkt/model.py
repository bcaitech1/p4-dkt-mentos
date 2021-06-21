import torch
import torch.nn as nn
import numpy as np
import re
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        
        cate_size = len(self.args.cate_cols) + 1 # interaction

        #1 Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_cols['testId']+1, self.args.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_cols['assessmentItemID']+1, self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_cols['KnowledgeTag']+1, self.args.hidden_dim)
        self.embedding_time_diff = nn.Embedding(self.args.n_cols['time_diff']+1, self.args.hidden_dim)
        self.embedding_head = nn.Embedding(self.args.n_cols['head']+1, self.args.hidden_dim)
        self.embedding_mid = nn.Embedding(self.args.n_cols['mid']+1, self.args.hidden_dim)
        self.embedding_tail = nn.Embedding(self.args.n_cols['tail']+1, self.args.hidden_dim)
        self.embedding_mid_tail = nn.Embedding(self.args.n_cols['mid_tail']+1, self.args.hidden_dim)

        # embedding projection
        self.cate_proj = nn.Sequential(
            nn.Linear((self.args.hidden_dim) * (cate_size), self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
        )

        self.lstm = nn.LSTM(self.args.hidden_dim,
                            self.args.hidden_dim,
                            self.args.n_layers,
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

        def get_reg():
            return nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
            nn.Dropout(self.args.drop_out),
            nn.ReLU(),            
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
        )     

        self.reg_layer = get_reg()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.args.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.args.device)

        return (h, c)

    def forward(self, inputs):

        #2 process_batch의 return 
        # tail_prob
        (test, question, tag, time_diff, 
        head, mid, tail, mid_tail, 
        head_answerProb, mid_answerProb, tail_answerProb,
        mask, interaction, gather_index, correct) = inputs

        batch_size = interaction.size(0)

        # Embedding

        #3 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_time_diff = self.embedding_time_diff(time_diff)
        embed_head = self.embedding_head(head)
        embed_mid = self.embedding_mid(mid)
        embed_tail = self.embedding_tail(tail)
        embed_mid_tail = self.embedding_mid_tail(mid_tail)
        

        #4
        # (batch_size, max_seq_len, hidden_dim * n) : (64, 20, 64 * n) 
        cate_embed = torch.cat([
                    embed_interaction,
                    embed_question,
                    embed_time_diff,
                    embed_tag,
                    embed_head,
                    embed_mid,
                    embed_tail,
                    embed_test,
                    embed_mid_tail,
                ], 2)

        cate_X = self.cate_proj(cate_embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(cate_X, hidden)
        # out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
        out = self.reg_layer(out)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):
    
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.args.device = args.device

        cate_size = len(self.args.cate_cols) + 1 # interaction

        #1 Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_cols['testId']+1, self.args.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_cols['assessmentItemID']+1, self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_cols['KnowledgeTag']+1, self.args.hidden_dim)
        self.embedding_time_diff = nn.Embedding(self.args.n_cols['time_diff']+1, self.args.hidden_dim)
        self.embedding_head = nn.Embedding(self.args.n_cols['head']+1, self.args.hidden_dim)
        self.embedding_mid = nn.Embedding(self.args.n_cols['mid']+1, self.args.hidden_dim)
        self.embedding_tail = nn.Embedding(self.args.n_cols['tail']+1, self.args.hidden_dim)
        self.embedding_mid_tail = nn.Embedding(self.args.n_cols['mid_tail']+1, self.args.hidden_dim)

       # embedding projection
        self.cate_proj = nn.Sequential(
            nn.Linear((self.args.hidden_dim) * (cate_size), self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
        )

        self.lstm = nn.LSTM(self.args.hidden_dim,
                            self.args.hidden_dim,
                            self.args.n_layers,
                            batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.args.n_heads,
            intermediate_size=self.args.hidden_dim,
            hidden_drop_out_prob=self.args.drop_out,
            attention_probs_drop_out_prob=self.args.drop_out,
        )
        self.attn = BertEncoder(self.config)            
    
        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

        def get_reg():
            return nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
            nn.Dropout(self.args.drop_out),
            nn.ReLU(),            
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
        )     

        self.reg_layer = get_reg()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.args.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.args.device)

        return (h, c)

    def forward(self, inputs):

        #2 process_batch의 return 
        # tail_prob
        (test, question, tag, time_diff, 
        head, mid, tail, mid_tail, 
        head_answerProb, mid_answerProb, tail_answerProb,
        mask, interaction, gather_index, correct) = inputs

        batch_size = interaction.size(0)

        # Embedding

        #3 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_time_diff = self.embedding_time_diff(time_diff)
        embed_head = self.embedding_head(head)
        embed_mid = self.embedding_mid(mid)
        embed_tail = self.embedding_tail(tail)
        embed_mid_tail = self.embedding_mid_tail(mid_tail)
        

        #4
        # (batch_size, max_seq_len, hidden_dim * n) : (64, 20, 64 * n) 
        cate_embed = torch.cat([
                    embed_interaction,
                    embed_question,
                    embed_time_diff,
                    embed_tag,
                    embed_head,
                    embed_mid,
                    embed_tail,
                    embed_test,
                    embed_mid_tail,
                ], 2)

        cate_X = self.cate_proj(cate_embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(cate_X, hidden)
        out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.args.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        #확인
        out = self.reg_layer(sequence_output)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

# feature 추가 #1~6 수정
class Bert(nn.Module):
    
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.args.numeric = False
        
        cate_size = len(self.args.cate_cols) + 1 # interaction
        cont_size = len(self.args.cont_cols) - 2


        #1 Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_cols['testId']+1, self.args.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_cols['assessmentItemID']+1, self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_cols['KnowledgeTag']+1, self.args.hidden_dim)
        self.embedding_time_diff = nn.Embedding(self.args.n_cols['time_diff']+1, self.args.hidden_dim)
        self.embedding_head = nn.Embedding(self.args.n_cols['head']+1, self.args.hidden_dim)
        self.embedding_mid = nn.Embedding(self.args.n_cols['mid']+1, self.args.hidden_dim)
        self.embedding_tail = nn.Embedding(self.args.n_cols['tail']+1, self.args.hidden_dim)
        self.embedding_mid_tail = nn.Embedding(self.args.n_cols['mid_tail']+1, self.args.hidden_dim)


        # embedding projection
        self.cate_proj = nn.Sequential(
            nn.Linear((self.args.hidden_dim) * (cate_size), self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
        )

        if self.args.numeric:
            self.cont_bn = nn.BatchNorm1d(cont_size)
            self.cont_proj = nn.Sequential(
                nn.Linear(cont_size, self.args.hidden_dim),
                nn.LayerNorm(self.args.hidden_dim),
            )

            self.comb_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim),
                nn.LayerNorm(self.args.hidden_dim),
            )

        else:
            self.comb_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
                nn.LayerNorm(self.args.hidden_dim),
            )

        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            intermediate_size=self.args.hidden_dim,
            hidden_drop_out_prob=self.args.drop_out,
            attention_probs_drop_out_prob=self.args.drop_out,
            max_position_embeddings=self.args.max_seq_len           
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()

        def get_reg():
            return nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
            nn.Dropout(self.args.drop_out),
            nn.ReLU(),            
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
        )     

        self.reg_layer = get_reg()


    def forward(self, inputs):
        #2 process_batch의 return 
        # tail_prob
        (test, question, tag, time_diff, 
        head, mid, tail, mid_tail, 
        head_answerProb, mid_answerProb, tail_answerProb,
        mask, interaction, gather_index, correct) = inputs

        batch_size = interaction.size(0)

        #3 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_time_diff = self.embedding_time_diff(time_diff)
        embed_head = self.embedding_head(head)
        embed_mid = self.embedding_mid(mid)
        embed_tail = self.embedding_tail(tail)
        embed_mid_tail = self.embedding_mid_tail(mid_tail)

        #4
        # (batch_size, max_seq_len, hidden_dim * n) : (64, 20, 64 * n) 
        cate_embed = torch.cat([
                    embed_interaction,
                    embed_question,
                    embed_time_diff,
                    embed_tag,
                    embed_head,
                    embed_mid,
                    embed_tail,
                    embed_test,
                    embed_mid_tail,
                ], 2)

        #5
        # (batch_size, max_seq_len * n) : (64, 20 * n) 
        if self.args.numeric:
            cont_cat = torch.cat([
                    # time_diff,  
                    # head_answerProb,   
                    # mid_answerProb,
                    tail_answerProb,
            ], 1)
            
            # (batch_size, max_seq_len, n) : (64, 20, n) 
            cont_cat = cont_cat.view(batch_size, self.args.max_seq_len, -1)

            # (batch_size * max_seq_len, n) : (1280, 1)
            cont_bn_x = self.cont_bn(cont_cat.view(-1, cont_cat.size(-1)))

            # (batch_size, max_seq_len, n) : (64, 20, n) 
            cont_bn_x = cont_bn_x.view(batch_size, self.args.max_seq_len, -1) 

        # (batch_size, max_seq_len, hidden_dim) : (64, 20, 64)
        cate_X = self.cate_proj(cate_embed)

        if self.args.numeric:
            cont_X = self.cont_proj(cont_cat)

            # (batch_size, max_seq_len, hidden_dim * 2) : [64, 20, 128]
            X = torch.cat([cate_X, cont_X], 2)
            

        else:
            X = cate_X

        # (batch_size, max_seq_len, hidden_dim) :[64, 20, 64]
        comb_X = self.comb_proj(X)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=comb_X, attention_mask=mask)
        out = encoded_layers[0]
        
        
        # base
        # out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
        # out = self.fc(out)
        # preds = self.activation(out).view(batch_size, -1)

        # + reg_layer
        out = self.reg_layer(out)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        self.args = args
        self.args.nn_Dropout = nn.Dropout(p=self.args.drop_out)
        self.scale = nn.Parameter(torch.ones(1))
        d_model = self.args.hidden_dim
        max_len = self.args.max_seq_len
        
    
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.args.nn_Dropout(x)


class Saint(nn.Module):
    
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        cate_size = len(self.args.cate_cols)
        cont_size = len(self.args.cont_cols)

        ### Embedding 
        #1 Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_cols['testId']+1, self.args.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_cols['assessmentItemID']+1, self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_cols['KnowledgeTag']+1, self.args.hidden_dim)
        self.embedding_time_diff = nn.Embedding(self.args.n_cols['time_diff']+1, self.args.hidden_dim)
        self.embedding_head = nn.Embedding(self.args.n_cols['head']+1, self.args.hidden_dim)
        self.embedding_mid = nn.Embedding(self.args.n_cols['mid']+1, self.args.hidden_dim)
        self.embedding_tail = nn.Embedding(self.args.n_cols['tail']+1, self.args.hidden_dim)
        self.embedding_mid_tail = nn.Embedding(self.args.n_cols['mid_tail']+1, self.args.hidden_dim)

        # embedding projection
        self.cate_proj = nn.Sequential(
            nn.Linear((self.args.hidden_dim) * (cate_size), self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
        )

        # encoder combination projection
        self.enc_comb_proj = nn.Sequential(
            nn.Linear((self.args.hidden_dim) * cate_size, self.args.hidden_dim),
            # nn.LayerNorm(self.args.hidden_dim),
        )

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        
        # decoder combination projection
        self.dec_comb_proj = nn.Sequential(
            nn.Linear((self.args.hidden_dim) * (cate_size + 1), self.args.hidden_dim),
            # nn.LayerNorm(self.args.hidden_dim),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(args)
        self.pos_decoder = PositionalEncoding(args)
        

        self.transformer = nn.Transformer(
            d_model=self.args.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.args.hidden_dim, 
            dropout=self.args.drop_out, 
            activation='relu')

        self.fc = nn.Linear(self.args.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

        def get_reg():
            return nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
            nn.Dropout(self.args.drop_out),
            nn.ReLU(),            
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
        )     

        self.reg_layer = get_reg()
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, inputs):
        #2 process_batch의 return 
        # tail_prob
        (test, question, tag, time_diff, 
        head, mid, tail, mid_tail, 
        head_answerProb, mid_answerProb, tail_answerProb,
        mask, interaction, gather_index, correct) = inputs

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        #3 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_time_diff = self.embedding_time_diff(time_diff)
        embed_head = self.embedding_head(head)
        embed_mid = self.embedding_mid(mid)
        embed_tail = self.embedding_tail(tail)
        embed_mid_tail = self.embedding_mid_tail(mid_tail)

        embed_enc = torch.cat([
                    embed_question,
                    embed_time_diff,
                    embed_tag,
                    embed_head,
                    embed_mid,
                    embed_tail,
                    embed_test,
                    embed_mid_tail,
                ], 2)

        embed_enc = self.enc_comb_proj(embed_enc)
        
        # DECODER     
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_interaction = self.embedding_interaction(interaction)

        embed_dec = torch.cat([
                    embed_question,
                    embed_time_diff,
                    embed_tag,
                    embed_head,
                    embed_mid,
                    embed_tail,
                    embed_test,
                    embed_mid_tail,
                    embed_interaction,
                ], 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.args.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.args.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.args.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        # out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
        out = self.reg_layer(out)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class TfixupBert(nn.Module):
    def __init__(self, args):
        super(TfixupBert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.args.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_cols['testId'], self.args.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_cols['assessmentItemID'], self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_cols['KnowledgeTag'], self.args.hidden_dim)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim)*4, self.hidden_dim)

        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len          
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()
        
        # T-Fixup
        if True:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixupbb Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*Norm.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param   
            elif re.match(r'.*Norm.*', name):
                continue
            elif re.match(r'encoder.*dense.*weight$|encoder.*attention.output.*weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]
                
        self.load_state_dict(temp_state_dict)


    def forward(self, inputs):
        test, question, tag, mask, interaction, gather_index, correct = inputs
        
        batch_size = interaction.size(0)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           ], 2)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds