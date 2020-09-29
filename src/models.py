#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')

from absl import logging,flags
import torch 
import numpy as np
import math
import IPython
import torch.nn.functional as F
from matplotlib import pyplot as plt

flags.DEFINE_integer('hidden_size', 128, '')
flags.DEFINE_integer('n_layers', 3, '')
flags.DEFINE_integer('n_layers_decoder', 1, '')
flags.DEFINE_float('dropout', 0.5, '')
flags.DEFINE_integer('n_pos', 32, '')
flags.DEFINE_bool('use_bahdanau_attention', True, '')
flags.DEFINE_bool('convolve_eeg_1d', False, '')

flags.DEFINE_bool('convolve_eeg_2d', False, '')

flags.DEFINE_bool('convolve_eeg_3d', False, '')
flags.DEFINE_bool('pre_and_postnet', True, '')
flags.DEFINE_integer('pre_and_postnet_dim', 256, '')

FLAGS = flags.FLAGS

def create_attention_plot(attention_matrix):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax.imshow(attention_matrix.detach().cpu().numpy(), cmap='viridis',aspect='auto') 
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #return torch.from_numpy(data.transpose(1,0,2)).float() / 255
    return torch.from_numpy(data).float() / 255
class PositionalEncoding(torch.nn.Module):

    def __init__(self, max_length, dim):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=FLAGS.dropout)

        pe = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1) # [max_length x 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)) # [dim/2 x 1]
        pe[:, 0::2] = torch.sin(position * div_term) #all even cols
        pe[:, 1::2] = torch.cos(position * div_term) #all odd cols
        self.register_buffer('pe', pe) # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.

    def forward(self, x):
        pos = self.pe[:x.shape[1]]
        pos = torch.stack([pos]*x.shape[0], 0) # [bs x seq_len(x) x n_pos]
        x = torch.cat((x, pos), -1)
        return self.dropout(x)

#########################
#### ATTENTION STUFF ####
#########################

class LocationLayer(torch.nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = self.location_dense(processed_attention.transpose(1, 2))
        return processed_attention

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Hybdrid_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_layer = LinearNorm(FLAGS.hidden_size, FLAGS.hidden_size,bias=False,w_init_gain='tanh')
        self.v = LinearNorm(FLAGS.hidden_size, 1, bias=False) 
        self.memory_layer = LinearNorm(FLAGS.hidden_size * 2, FLAGS.hidden_size,bias=False,w_init_gain='tanh') #V in Bahdanau
        self.location_layer = LocationLayer(32,31,FLAGS.hidden_size) #FLAGS.hidden_size = attention_dim
  
    def compute_kv(self,encoder_outputs):
        return self.memory_layer(encoder_outputs)  

    def compute_context(self,inp_emb,hiddens,attn_kv,attention_weights_cat):
        attn_q= self.query_layer(hiddens[-1]).unsqueeze(1) #Ws. [bs x 1 x hidden_size] using only the hidden states of the top layer as query. 
        attn_locs=self.location_layer(attention_weights_cat) #attention_weights_cat= previous and cummulative attention weights
        attn_scores=torch.softmax(self.v(torch.tanh(attn_q+attn_kv+attn_locs)), dim=1).squeeze(-1) # [bs x seq_len_input]
        return torch.bmm(attn_scores.unsqueeze(1), attn_kv), attn_scores.unsqueeze(1) # context shape: [bs, 1, 1* hidden_dim] -> attend to linear comb of encoder directions 

class Yannic_Attention(torch.nn.Module):
    def __init__(self,embedding_size):
        super().__init__()

        self.attention_q = torch.nn.Linear(FLAGS.hidden_size * FLAGS.n_layers_decoder + embedding_size + FLAGS.n_pos, FLAGS.hidden_size)
        self.attention_kv = torch.nn.Linear(FLAGS.hidden_size * 2, FLAGS.hidden_size) #V in Bahdanau

    def compute_kv(self,encoder_outputs):
        return self.attention_kv(encoder_outputs)  

    def compute_context(self,inp_emb,hiddens,attn_kv):

        attn_q = self.attention_q(torch.cat((inp_emb, hiddens.transpose(1, 0).flatten(1).unsqueeze(1)), -1))# [bs x 1 x hidden_size]
        # input is [bs x 1 x emb_dim/mel_bins + num_pos + hidden_dim * num_layers]  not just hidden state but also input of decoder are used as query
        # output is [bs x 1 x hidden_dim]
        attn_scores = F.softmax(torch.bmm(attn_q, attn_kv.transpose(2, 1)), -1) # return [bs x 1 x seq_len_input]
        #bmm computes for each sample: the vector matrix multiplication of attn_q [1 x hidd] x attn_kv.T [hidd x seq_len_inp] 
        # = (<hidden_enc, hidden_dec_1>, ,<hidden_enc, hidden_dec_2>, ... ,  <hidden_enc, hidden_dec_t>) 
        # this gives the attention scores for each point in time, passed through a softmax for normalized weights. 

        return torch.bmm(attn_scores, attn_kv),attn_scores # [bs x 1 x hidden_dim] #The final attention vector is obtained by multiplying the scores with the attention key values. ENC HIDDEN STATES in bahdanau!

#########################
#########################
#########################    

class Prenet(torch.nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = torch.nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.elu(linear(x)), p=FLAGS.dropout, training=True)
        return x


class Postnet(torch.nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = torch.nn.ModuleList()
        self.postnet_embedding_dim=FLAGS.pre_and_postnet_dim # out channels. 
        self.postnet_kernel_size=5
        self.n_mel_channels=80 # in channels
        if FLAGS.discretize_MFCCs:
             self.n_mel_channels=self.n_mel_channels*FLAGS.num_mel_centroids
        self.postnet_n_convolutions=5
        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(self.n_mel_channels, self.postnet_embedding_dim,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                torch.nn.BatchNorm1d(self.postnet_embedding_dim))
        )

        for i in range(1, self.postnet_n_convolutions - 1):
            self.convolutions.append(
                torch.nn.Sequential(
                    ConvNorm(self.postnet_embedding_dim,
                             self.postnet_embedding_dim,
                             kernel_size=self.postnet_kernel_size, stride=1,
                             padding=int((self.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    torch.nn.BatchNorm1d(self.postnet_embedding_dim))
            )

        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(self.postnet_embedding_dim, self.n_mel_channels,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                torch.nn.BatchNorm1d(self.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), FLAGS.dropout, self.training)
        x = F.dropout(self.convolutions[-1](x), FLAGS.dropout, self.training)

        return x          

class Convnet(torch.nn.Module):
    def __init__(self, input_channels, conv_channels_1=50,conv_channels_2=75):
        super(Convnet, self).__init__()
#original:         self.conv1=ConvNorm(in_channels=input_channels, out_channels=conv_channels_1, kernel_size=25, stride=3,w_init_gain='relu')         # (N,500,)


        self.conv1=ConvNorm(in_channels=input_channels, out_channels=conv_channels_1, kernel_size=25, stride=3,w_init_gain='relu')         # (N,500,)
        self.bn1=torch.nn.BatchNorm1d(num_features=conv_channels_1)
        self.conv2=ConvNorm(in_channels=conv_channels_1, out_channels=conv_channels_2, kernel_size=11, stride=3,w_init_gain='relu')                  # (N,500,)
        self.bn2=torch.nn.BatchNorm1d(num_features=conv_channels_2)        
        self.conv3=ConvNorm(in_channels=conv_channels_2, out_channels=100, kernel_size=5, stride=3,w_init_gain='relu')                  # (N,500,)
        self.bn3=torch.nn.BatchNorm1d(num_features=100)            
        self.maxp=torch.nn.MaxPool1d(kernel_size=2, stride=2)   #in total x is downscaled by a factor of 9
        self.act=torch.nn.ELU()
    def forward(self, x):
        x=self.act(self.bn3(self.conv3(self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))))))
        return self.maxp(x) 


class Convnet2d(torch.nn.Module):
    def __init__(self, input_channels, conv_channels_1=25,conv_channels_2=50):
        super(Convnet2d, self).__init__()

        self.conv1=torch.nn.Conv2d(in_channels=1, out_channels=conv_channels_1, kernel_size=25, stride=2)         # (N,500,)
        self.bn1=torch.nn.BatchNorm2d(num_features=conv_channels_1)
        self.conv2=torch.nn.Conv2d(in_channels=conv_channels_1, out_channels=conv_channels_2, kernel_size=12, stride=3)                  # (N,500,)
        self.bn2=torch.nn.BatchNorm2d(num_features=conv_channels_2)          
        self.conv3=torch.nn.Conv2d(in_channels=conv_channels_2, out_channels=75, kernel_size=8, stride=2)                  # (N,500,)
        self.bn3=torch.nn.BatchNorm2d(num_features=75)          
        self.maxp=torch.nn.MaxPool2d(kernel_size=2, stride=2)   #in total x is downscaled by a factor of 9
        self.act=torch.nn.ELU()
    def forward(self, x):
        x=x.unsqueeze(1) # bs x eeg_channels x seq_len -> bs x conv_in_channels x eeg_channels x seq_len
        x=self.act(self.bn3(self.conv3(self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))))))
        return self.maxp(x).flatten(1,2)


class Convnet3d(torch.nn.Module):
    def __init__(self, input_channels, conv_channels_1=15,conv_channels_2=30):
        super(Convnet3d, self).__init__()

        self.conv1=torch.nn.Conv3d(in_channels=1, out_channels=conv_channels_1, kernel_size=3, stride=2,padding=3)         # (N,500,)
        self.bn1=torch.nn.BatchNorm3d(num_features=conv_channels_1)
        self.conv2=torch.nn.Conv3d(in_channels=conv_channels_1, out_channels=conv_channels_2, kernel_size=3, stride=2,padding=3)                  # (N,500,)
        self.bn2=torch.nn.BatchNorm3d(num_features=conv_channels_2)          
        self.maxp=torch.nn.MaxPool3d(kernel_size=3, stride=2,padding=1)   #in total x is downscaled by a factor of 9
        self.act=torch.nn.ELU()
    def forward(self, x):
        x=x.unsqueeze(1)  #bs x 1 x seq_len x 9 x 12 (1 channel for conv)
        x=self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))) #bs x 30 x 53 x 6 x 6
        x= self.maxp(x) #bs x 30 (out_channels) x 27 (seq_len) x 3 x 3
        x=x.transpose(1,2) #bs x seq_len x channels x 3 x3 
        x=x.flatten(2,4) #bs x seq_len x 270
        return x

#########################
#########################
#########################   


class RNNSeq2Seq(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        assert not (FLAGS.convolve_eeg_1d and FLAGS.convolve_eeg_2d) and not (FLAGS.convolve_eeg_2d and FLAGS.convolve_eeg_3d) and not (FLAGS.convolve_eeg_1d and FLAGS.convolve_eeg_3d), "inupt can only be convolved by 1d OR 2d filters"

        self.input_length, self.input_channels = input_shape #seq_len_input, num_channels
        self.output_length, self.num_classes = output_shape #seq_len_output, num_classes/num_mel_bins
        if FLAGS.discretize_MFCCs: 
            self.num_classes=self.num_classes*FLAGS.num_mel_centroids 
        self.embedding_size = self.num_classes if FLAGS.use_MFCCs else FLAGS.hidden_size
        self.input_position_encoder = PositionalEncoding(self.input_length, FLAGS.n_pos)
        self.output_position_encoder = PositionalEncoding(self.output_length, FLAGS.n_pos) 

        self.context_for_prediction=True


        self.input_dim=self.input_channels

        if FLAGS.convolve_eeg_2d:
            self.convnet=Convnet2d(self.input_channels)
            self.input_dim=75
        elif FLAGS.convolve_eeg_1d:
            self.convnet=Convnet(self.input_channels)
            self.input_dim=self.convnet.conv3.conv.out_channels
        elif FLAGS.convolve_eeg_3d:
            self.convnet=Convnet3d(self.input_channels)
            self.input_dim=270

        self.encoder = torch.nn.GRU( 
                self.input_dim + FLAGS.n_pos, # input_size, positional embeddings are added to the channels. CONV CHANNELS + pos IF convolve_first.
                hidden_size=FLAGS.hidden_size, 
                num_layers=FLAGS.n_layers,
                bidirectional=True,
                batch_first=True,
                dropout=FLAGS.dropout
                )

        self.hidden_to_hidden = torch.nn.Linear(FLAGS.hidden_size*FLAGS.n_layers*2, FLAGS.hidden_size*FLAGS.n_layers_decoder)

        if not FLAGS.use_MFCCs:
            self.audio_embedding = torch.nn.Embedding(self.num_classes+1, self.embedding_size) # num_embeddings, embedding_dim


        prenet_dim=FLAGS.pre_and_postnet_dim  
        if FLAGS.discretize_MFCCs:
            self.prenet=Prenet(80,[prenet_dim, prenet_dim]) # num mel bins (num_classes now mel_bins*num_centroids, hidden layer 1 size, hidden layer 2 size
        else:
            self.prenet=Prenet(self.num_classes,[prenet_dim, prenet_dim]) # num mel bins, hidden layer 1 size, hidden layer 2 size

        if FLAGS.pre_and_postnet:
            self.embedding_size=prenet_dim
        self.decoder = torch.nn.GRU(
                FLAGS.hidden_size + self.embedding_size + FLAGS.n_pos,
                hidden_size=FLAGS.hidden_size,
                num_layers=FLAGS.n_layers_decoder,
                bidirectional=False,
                batch_first=True
                )

        if self.context_for_prediction:
            #self.decoder_classifier = LinearNorm(2*FLAGS.hidden_size, self.num_classes)
            if FLAGS.discretize_MFCCs:
                self.decoder_classifier = torch.nn.Sequential(LinearNorm(2*FLAGS.hidden_size, 3*self.num_classes),torch.nn.ELU(),
                torch.nn.Dropout(FLAGS.dropout),LinearNorm(3*self.num_classes, self.num_classes))
            else:
                self.decoder_classifier = torch.nn.Sequential(LinearNorm(2*FLAGS.hidden_size, FLAGS.hidden_size),torch.nn.ELU(),
                torch.nn.Dropout(FLAGS.dropout),LinearNorm(FLAGS.hidden_size, self.num_classes))


        else:
            self.decoder_classifier = LinearNorm(FLAGS.hidden_size, self.num_classes)

        self.postnet=Postnet()

        if FLAGS.use_bahdanau_attention:
            self.attention=Hybdrid_Attention()
        else:
            self.attention=Yannic_Attention(self.embedding_size)



    def forward(self, x, y=None, teacher_forcing=None):
        assert (y is None) == (teacher_forcing is None)

        #x is in bs x seq_len x n_channels

        flip=True ### reverse time order of input for the encoder.
        if flip:
            x=torch.flip(x,[2])

        if FLAGS.convolve_eeg_3d:
            extra_row=torch.zeros_like(x[:,:,0])  
            x=torch.cat((x.T,extra_row.unsqueeze(-1).T)).T
            x=x.view(-1,self.input_length,9,12) #bs x seq_len x 9 x 12
            x=self.convnet(x) #returns bs x seq_len x channels (270)

        elif (FLAGS.convolve_eeg_1d or FLAGS.convolve_eeg_2d): 
            x=self.convnet(x.transpose(1,2)).transpose(1,2) #x is now in bs x seq_len x channels

        encoder_outputs, hiddens = self.encoder(self.input_position_encoder(x)) # returns [batch_size, seq_len_input,hidden_size * num_directions] & [num_layers * num_directions, batch_size, hidden_dim]

        # context vector (first input to decoder):
        hiddens = self.hidden_to_hidden(hiddens.transpose(1, 0).flatten(1)).reshape((-1, FLAGS.n_layers_decoder, FLAGS.hidden_size)).transpose(1, 0).contiguous()
        # first concatenates hidden states of all encoder layers and directions, projects onto [bs, hidden states * num_layers] with linear layer, reshapes to [num_layers x bs x hidden states]
        # alternative: just use hidden states of forward direction.
        attn_kv=self.attention.compute_kv(encoder_outputs) # [batch_size x seq_len_input x 2* hidden_dim] to [batch_size x seq_len_input x hidden_dim]

        inp = x.new_zeros((x.shape[0], 1,self.num_classes)) if FLAGS.use_MFCCs else x.new_ones((x.shape[0], 1)).long() * self.num_classes # SOS
        if FLAGS.discretize_MFCCs:
            inp = x.new_zeros((x.shape[0], 1,80))
        if FLAGS.pre_and_postnet:
            inp = self.prenet(inp)

        logits = []
        attn_matrix=[]
        attn_scores=encoder_outputs.data.new_zeros((encoder_outputs.shape[0],1,encoder_outputs.shape[1]))
        attn_scores_cum=torch.zeros_like(attn_scores)
        for i in range(self.output_length):

            inp_emb = inp if FLAGS.use_MFCCs else self.audio_embedding(inp) # [bs x 1 x hidden_size (emd dim)]
            inp_emb = self.output_position_encoder(inp_emb) # [bs x 1 x  emd dim/mel bins + n_pos]. 

            attention_scores_cat = torch.cat((attn_scores,attn_scores_cum), dim=1)
            if FLAGS.use_bahdanau_attention:
                attn_context,attn_scores=self.attention.compute_context(inp_emb,hiddens,attn_kv,attention_scores_cat)
            else:
                attn_context,attn_scores=self.attention.compute_context(inp_emb,hiddens,attn_kv)

            attn_scores_cum+= attn_scores
            attn_matrix.append(attn_scores[0]) #just take first sample in batch

            decoder_input=torch.cat((inp_emb, attn_context), -1) # [bs x 1 x  emd dim/mel bins + n_pos + hidden_size ].  (hidden_size=attention_dim for us)
            decoder_outputs, hiddens = self.decoder(decoder_input, hiddens)

            if self.context_for_prediction:
                decoder_outputs=torch.cat([decoder_outputs, attn_context], 2) # [bs x 1 x 2*hidden]
            else:
                decoder_outputs=decoder_outputs # [bs x 1 x hidden]

            l = self.decoder_classifier(decoder_outputs) # [bs x 1 x num_classes/mel_bins]
          
            if FLAGS.use_MFCCs and FLAGS.pre_and_postnet:
                # 1. Pass the mel prediction through the postnet and combine them for the final prediction
                mel_outputs_postnet = self.postnet(l.transpose(1,2)).transpose(1,2)
                final_mel_output = l + mel_outputs_postnet
                logits.append(final_mel_output)

            else:
                logits.append(l)

            # 2. improve prediction with TF
            if FLAGS.discretize_MFCCs:
                #convert logits to classes
                current_batch_size=l.shape[0] #will be flags.batchsize almost always but not in the very end of validation/train
                l=l.reshape((current_batch_size,-1,80,FLAGS.num_mel_centroids)) # bs x seq_len x 80 x 12
                l=torch.argmax(l,dim=3) #bs x seq_len x 80. each entry is the class.

            if FLAGS.use_MFCCs:
                if teacher_forcing is not None:
                    l = torch.where(teacher_forcing.repeat(y.shape[2],1).T, y[:, i,:], l.squeeze(1)).unsqueeze(1) # y is  ([bs, seq_len, mel_bins])
                inp=l
            else:
                _, topi = l.topk(k=1, dim=-1) #@YK: why not use max?
                topi = topi.detach().squeeze()
                if teacher_forcing is not None:
                    topi = torch.where(teacher_forcing, y[:, i], topi) #replace prediction with ground truth for all samples with tf true
                inp=topi.unsqueeze(1)

            # 3. Pass the prediction through the prenet before feeding it back to the RNN 
            if FLAGS.pre_and_postnet:
                inp=self.prenet(inp.float()) # [bs x 1 x 256]


        attn_matrix=torch.cat(attn_matrix)
        logits = torch.cat(logits, 1)
        return logits,attn_matrix, encoder_outputs

class OLSModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.input_length,self.input_dim = input_shape
        self.output_length, self.num_classes = output_shape
        self.classifier = torch.nn.Linear(self.input_length*self.input_dim, self.num_classes)

    def forward(self, x, y=None, teacher_forcing=None):
        prediction=self.classifier(x.view((x.shape[0],x.shape[1]*x.shape[2]))) # x from [bs x inp_seq_len x num_channels] to [bs x inp_seq_len * num_channels]
        return prediction.unsqueeze(1) #add time dimension


class DensenetModel(torch.nn.Module):
    def __init__(self,channels=40,new_channels_per_conv=20):
        super().__init__()
        self.initial_conv= torch.nn.Conv3d(in_channels=1, out_channels=channels, kernel_size=3, stride=1,padding=1,bias=False)
        #dim: out= (in +2*padding-kernel)/stride+1 --> padding (in!=out) = ((s-1)*in+kernel-1)/2 --> padding (stride!=1) = (kernel-1)/2

        self.dense_block_1= DenseBlock(channels_in=channels,new_channels_per_conv=new_channels_per_conv) #goes in as T,H,B,20. comes out as T,H,B,40 (2 conv layers add 10 filters each)
        channels+=2*new_channels_per_conv
        
        self.transition_1=TransitionBlock(channels_in=channels)
        self.dense_block_2=DenseBlock(channels_in=channels,new_channels_per_conv=new_channels_per_conv)
        channels+=2*new_channels_per_conv

        self.transition_2=TransitionBlock(channels_in=channels)
        self.dense_block_3=DenseBlock(channels_in=channels,new_channels_per_conv=new_channels_per_conv)
        channels+=2*new_channels_per_conv

        self.final_bn=torch.nn.BatchNorm3d(num_features=channels)
        self.classifier=torch.nn.Linear(channels,80) #should be 160 to 80
        if FLAGS.discretize_MFCCs:
            #self.classifier = torch.nn.Linear(channels,self.num_classes*FLAGS.num_mel_centroids)
            self.classifier= torch.nn.Sequential(LinearNorm(channels,3*channels),torch.nn.ELU(),LinearNorm(3*channels, 80*FLAGS.num_mel_centroids))

    def forward(self,x,y=None, teacher_forcing=None):
        #only gamma
        extra_row=torch.zeros_like(x[:,:,0])  
        x=torch.cat((x.T,extra_row.unsqueeze(-1).T)).T
        x=x.view(-1,x.shape[1],9,12) #bs x seq_len x 9 x 12


        x=x.unsqueeze(1) #create fake channel.  #bs x 1 x seq_len x 9 x 12
        x=self.initial_conv(x)  #bs x 40 x seq_len x 9 x 12
        x=self.dense_block_1(x)  #bs x 80 x seq_len x 9 x 12
        x=self.transition_1(x) #bs x 80 
        x=self.dense_block_2(x) #bs x 120 
        x=self.transition_2(x) #bs x 120 x 13 x 3 x 3
        x=self.dense_block_3(x)#bs x 160 x 13 x 3 x 3
        x=torch.nn.functional.relu(self.final_bn(x))
        x=torch.mean(x,(2,3,4)) #for each channel, take the global mean of the 3d representation. -> x no in bs x 160
        return self.classifier(x).unsqueeze(1), x.new_zeros((10,10)), None #re-add time dim, fake attn matrix and enc outputs


class TransitionBlock(torch.nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.bn=torch.nn.BatchNorm3d(num_features=channels_in)
        self.act=torch.nn.ReLU()
        self.conv=torch.nn.Conv3d(in_channels=channels_in, out_channels=channels_in, kernel_size=1, stride=1,padding=0,bias=False)
        self.dropout = torch.nn.Dropout(p=FLAGS.dropout)
        self.avg_pooling=torch.nn.AvgPool3d(kernel_size=(3,3,3), stride=(2,2,2),padding=1)  
    def forward(self,input):
        return self.avg_pooling(self.dropout(self.conv(self.act(self.bn(input)))))

class DenseBlock(torch.nn.Module):
    def __init__(self,channels_in,new_channels_per_conv):
        super().__init__()

        self.act=torch.nn.ReLU()
        self.bn1=torch.nn.BatchNorm3d(num_features=channels_in)
        self.conv_1=torch.nn.Conv3d(in_channels=channels_in, out_channels=new_channels_per_conv, kernel_size=3, stride=1,padding=1,bias=False)
        self.dropout = torch.nn.Dropout(p=FLAGS.dropout)

        self.bn2=torch.nn.BatchNorm3d(num_features=channels_in+new_channels_per_conv)
        self.conv_2=torch.nn.Conv3d(in_channels=channels_in+new_channels_per_conv, out_channels=new_channels_per_conv, kernel_size=3, stride=1,padding=1,bias=False)

    def forward(self,input):
        self.feature_list=[]

        self.feature_list.append(input)
        # BatchNorm, Relu 3x3Conv2D, optional dropout
        x=self.dropout(self.conv_1(self.act(self.bn1(input))))
        self.feature_list.append(x)
        #concatenate with input
        x=torch.cat(self.feature_list, axis=1)

        #repeat:
        x=self.dropout(self.conv_2(self.act(self.bn2(x))))
        self.feature_list.append(x)
        x=torch.cat(self.feature_list, axis=1)

        return x
 


