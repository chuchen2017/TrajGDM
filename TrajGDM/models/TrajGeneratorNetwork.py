import torch
import torch.nn as nn
import math

class TrajGeneratorNetwork(torch.nn.Module):
    def __init__(self, num_location, location_embedding, lstm_hidden,maxi,device=torch.device("cuda:1"),input_len=12,num_head=2,TrajEncoder_layers=1,TrajGenerator_LSTMlayers=1,TrajGenerator_Translayers=1,TrajDecoder_layers=1):
        super(TrajGeneratorNetwork, self).__init__()
        self.num_location = num_location
        self.model_device=device
        self.input_len=input_len
        self.loc_size=location_embedding
        self.expand_num_loc = num_location + maxi*4
        self.num_head = num_head

        self.temporal_embed = TimeEmbedding(location_embedding).double()
        self.time_embed = TimeEmbedding(location_embedding).double()
        self.pos_encoding1 = PositionalEncoding(num_features=location_embedding, dropout=0.1, max_len=self.input_len*2)
        self.pos_encoding2 = PositionalEncoding(num_features=lstm_hidden, dropout=0.1, max_len=self.input_len*2)
        self.pos_encoding3 = PositionalEncoding(num_features=lstm_hidden, dropout=0.1, max_len=self.input_len*2)
        self.pos_encoding4 = PositionalEncoding(num_features=lstm_hidden, dropout=0.1, max_len=self.input_len * 2)
        self.start_embed = torch.nn.Embedding(num_embeddings=2, embedding_dim=location_embedding)

        self.location_embed = torch.nn.Embedding(num_embeddings=self.expand_num_loc, embedding_dim=self.loc_size)
        self.spatial_embed = nn.Linear(location_embedding*5, location_embedding)

        self.TrajEncoder_lstm = torch.nn.LSTM(input_size=self.loc_size, hidden_size=location_embedding, num_layers=TrajEncoder_layers,bidirectional=False, batch_first=True)

        self.layer_norm = nn.LayerNorm(location_embedding)
        self.layer_norm2 = nn.LayerNorm(lstm_hidden)
        self.layer_norm3 = nn.LayerNorm(lstm_hidden)
        self.layer_norm4 = nn.LayerNorm(lstm_hidden)

        self.TrajGenerator_encoderLSTM=torch.nn.LSTM(input_size=location_embedding, hidden_size=lstm_hidden, num_layers=TrajGenerator_LSTMlayers, bidirectional=False, batch_first=True)  #2

        self.TrajGenerator_encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=location_embedding, nhead=self.num_head,
                                                                                                  dim_feedforward=lstm_hidden, dropout=0.1,
                                                                                                  activation='relu', batch_first=True), num_layers=TrajGenerator_Translayers)

        self.TrajGenerator_decoder = torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=location_embedding, nhead=self.num_head,
                                                                                                  dim_feedforward=lstm_hidden, dropout=0.1,
                                                                                                  activation='relu', batch_first=True), num_layers=TrajGenerator_Translayers)

        self.TrajGenerator_decoderLSTM=torch.nn.LSTM(input_size=location_embedding, hidden_size=lstm_hidden, num_layers=1, bidirectional=False, batch_first=True)  #2

        self.TrajDecoder_encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=location_embedding, nhead=self.num_head,
                                                                                                dim_feedforward=lstm_hidden, dropout=0.1,
                                                                                                activation='relu', batch_first=True), num_layers=TrajDecoder_layers)

        self.TrajDecoder_decoder = torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=location_embedding, nhead=self.num_head,
                                                                                                dim_feedforward=lstm_hidden, dropout=0.1,
                                                                                                activation='relu', batch_first=True), num_layers=TrajDecoder_layers)

        self.fc_layer = torch.nn.Linear(in_features=lstm_hidden, out_features=lstm_hidden)
        self.determintric_function = nn.Linear(lstm_hidden, self.num_location)

    def TrajGenerator(self, xt, time):
        t = self.temporal_embed(time.double())
        xt_t = torch.unsqueeze(t, dim=1).repeat(1, xt.shape[1], 1)
        et = self.layer_norm(xt + xt_t)

        residual = et
        et, (hn, cn) = self.TrajGenerator_encoderLSTM(et)
        et = self.layer_norm2(et + residual)
        et = self.pos_encoding1(et)
        et = self.TrajGenerator_encoder(et)

        decoder_input = self.start_embed(torch.zeros(size=(et.shape[0], 1),device=et.device, dtype=torch.long))
        output=torch.zeros_like(xt,dtype=torch.double)

        for i in range(xt.shape[1]):
            if i == 0:
                decoder_input=self.pos_encoding2(decoder_input)
            else:
                previous=decoder_input[:,:-1,:]
                decoder_input = self.pos_encoding2(decoder_input)
                decoder_input[:,:-1,:]=previous
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(decoder_input.device).bool()
            decoder_output=self.TrajGenerator_decoder(decoder_input, et, tgt_mask)  #,tgt_mask
            output[:, i, :] = decoder_output[:, i, :]
            decoder_input = torch.cat((decoder_input,decoder_output[:,i,:].unsqueeze(1)),dim=1)
        output=self.fc_layer(output)
        return output

    def TrajDecoder(self, hidden_repr):
        src_mask = self.generate_square_subsequent_mask(hidden_repr.size(1)).to(hidden_repr.device).bool()
        hidden_repr = self.pos_encoding3(hidden_repr) # 1 3
        hidden_repr = self.TrajDecoder_encoder(hidden_repr, mask=src_mask)

        decoder_input = self.location_embed(torch.ones(size=(hidden_repr.shape[0], hidden_repr.shape[1]), device=hidden_repr.device,dtype=torch.long))

        decoder_input = self.pos_encoding4(decoder_input)  #2 4
        tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(decoder_input.device).bool()
        decoder_output = self.TrajDecoder_decoder(decoder_input, hidden_repr, tgt_mask)

        outputs = self.determintric_function(decoder_output)
        # nomiates = torch.argmax(outputs,dim=-1)
        return outputs  # ,nomiates

    def TrajEncoder(self,sequence):
        sequence, (hn, cn) = self.TrajEncoder_lstm(sequence)
        return sequence

    def LocationEncoder(self,locs,lab,maxi):
        if maxi == 110 or maxi == 27:
            west = self.location_embed(locs + maxi)
            east = self.location_embed(locs + 2 + maxi)
            north = self.location_embed(locs + 1)
            south = self.location_embed(locs + 1 + (2 * maxi))
            center = self.location_embed(locs + 1 + maxi)
            sequence = self.spatial_embed(torch.cat((center * lab, south, north, east, west), dim=-1))
            return sequence
        else:
            sequence = self.location_embed(locs)
            return sequence

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @property
    def get_device(self):
        return  self.model_device

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.dim=4
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class PositionalEncoding(nn.Module):
    def __init__(self, num_features, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_features))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_features, 2, dtype=torch.float32) / num_features)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
