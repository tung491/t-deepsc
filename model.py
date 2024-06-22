import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class PatchUnembedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchUnembedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = x.reshape(-1, self.embed_dim, self.grid_size, self.grid_size)  # (B, embed_dim, grid_size, grid_size)
        x = self.proj(x)  # (B, in_chans, img_size, img_size)
        return x


def dense(input_size, output_size):  # dense layer is a full connection layer and used to gather information
    return torch.nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.ReLU()
    )


class AWGNChannel(nn.Module):
    def __init__(self, snr: float = 12):
        super(AWGNChannel, self).__init__()
        self.snr = snr
        self.snr_factor = 10 ** (self.snr / 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the power of the input signal
        x_power = torch.mean(x ** 2)

        # Calculate the noise power based on SNR
        n_power = x_power / self.snr_factor

        # Generate Gaussian noise with the calculated noise power
        noise = torch.randn_like(x) * torch.sqrt(n_power)

        return x + noise

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class ViSemanticCommunicationSystem(nn.Module):  # pure DeepSC
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128, snr=12, K=8, noise_channel=AWGNChannel):
        super(ViSemanticCommunicationSystem, self).__init__()
        self.snr = snr
        self.K = K
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)  # Image patch embedding
        self.frontEncoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)  # according to the paper
        self.encoder = nn.TransformerEncoder(self.frontEncoder, num_layers=3)
        self.denseEncoder1 = dense(embed_dim, 256)
        self.denseEncoder2 = dense(256, 2 * self.K)
        self.noiseChannel = torch.jit.script(noise_channel(snr))

        self.denseDecoder1 = dense(2 * self.K, 256)
        self.denseDecoder2 = dense(256, embed_dim)
        self.frontDecoder = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(self.frontDecoder, num_layers=3)

        self.patch_unembedding = PatchUnembedding(img_size, patch_size, in_chans, embed_dim)

    def forward(self, inputs):
        embeddingVector = self.patch_embedding(inputs)  # Patch embedding
        code = self.encoder(embeddingVector)
        denseCode = self.denseEncoder1(code)
        codeSent = self.denseEncoder2(denseCode)

        codeWithNoise = self.noiseChannel(codeSent)

        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        codeSemantic = self.decoder(codeReceived, code)
        reconstructed_image = self.patch_unembedding(codeSemantic)
        return reconstructed_image

    def set_snr(self, snr):
        self.noiseChannel = torch.jit.script(AWGNChannel(snr))



def embedding(input_size, output_size): # embedding layer, the former is the size of dic and
    return nn.Embedding(input_size, output_size)

def dense(input_size, output_size): # dense layer is a full connection layer and used to gather information
    return torch.nn.Sequential(
    nn.Linear(input_size, output_size),
    nn.ReLU()
    )


class TextSemanticCommunicationSystem(nn.Module): # pure DeepSC
    def init(self, input_size, output_size=128, snr=12, K=8, noise_channel=AWGNChannel):
        super(TextSemanticCommunicationSystem, self).init()
        self.snr = snr
        self.K = K
        self.embedding = embedding(input_size, output_size) # which means the corpus has input_size kinds of words and
        # each word will be coded with a output_size dimensions vector
        self.frontEncoder = nn.TransformerEncoderLayer(d_model=output_size, nhead=8) # according to the paper
        self.encoder = nn.TransformerEncoder(self.frontEncoder, num_layers=3)
        self.denseEncoder1 = dense(output_size, 256)
        self.denseEncoder2 = dense(256, 2 * self.K)
        self.noiseChannel = torch.jit.script(noise_channel(snr))
        self.denseDecoder1 = dense(2 * self.K, 256)
        self.denseDecoder2 = dense(256, output_size)
        self.frontDecoder = nn.TransformerDecoderLayer(d_model=output_size, nhead=8)
        self.decoder = nn.TransformerDecoder(self.frontDecoder, num_layers=3)

        self.prediction = nn.Linear(output_size, input_size)
        self.softmax = nn.Softmax(dim=2)  # dim=2 means that it calculates softmax in the feature dimension

    def forward(self, inputs):
        embeddingVector = self.embedding(inputs)
        code = self.encoder(embeddingVector)
        denseCode = self.denseEncoder1(code)
        codeSent = self.denseEncoder2(denseCode)
        codeWithNoise = self.noiseChannel(codeSent)
        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        codeSemantic = self.decoder(codeReceived, code)
        codeSemantic = self.prediction(codeSemantic)
        info = self.softmax(codeSemantic)
        return info


