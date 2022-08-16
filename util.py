import torch

def make_img(dloader,G):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # load model
    Generator = G

    # load real image
    dloader = iter(dloader)
    img = next(dloader)

    img = img.type(dtype)

    # encoded latent vector
    # mu, log_var = Encoder(img)
    # std = torch.exp(log_var / 2)
    # random_z = torch.randn(1, 512).type(dtype)
    # encoded_z = (random_z * std) + mu

    # run mapping
    #style_w = Mapping(noise_z)

    # generate fake image
    noise = torch.randn(1,512).type(dtype)
    test_img = Generator(img,noise)


    # [-1, 1] -> [0, 1]
    result_img = test_img / 2 + 0.5

    return result_img
