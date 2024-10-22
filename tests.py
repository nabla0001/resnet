from model import ResNet, ResidualBlock, PlainBlock, ZeroPadding, ZeroPaddingMaxPool, Conv1x1Projection
import torch

def test_plain_block():
    plain = PlainBlock(32, 32, 1)

    x = torch.ones(size=(1, 32, 32, 32))
    y = plain(x)

    assert y.shape == (1, 32, 32, 32)

    plain = PlainBlock(32, 32, 2)
    y = plain(x)

    assert y.shape == (1, 32, 16, 16)


def test_residual_block():
    # dimensions of x and residual output match
    residual = ResidualBlock(16, 16, 1, skip_connection=None)

    x = torch.ones(1, 16, 32, 32)
    y = residual(x)

    assert y.shape == (1, 16, 32, 32)

    # dimensions of x and residual output do not match
    residual = ResidualBlock(16, 32, 2, skip_connection=ZeroPadding)

    x = torch.ones(1, 16, 32, 32)
    y = residual(x)

    assert y.shape == (1, 32, 16, 16)

def test_resnet_zero_padding():
    resnet = ResNet(ResidualBlock, ZeroPadding, (2, 2, 2,), 10)

    x = torch.ones(size=(10, 3, 32, 32)) # (N, C, H, W)
    y = resnet(x)

    assert y.shape == (10,10)

def test_resnet_conv1x1_projection():
    resnet = ResNet(ResidualBlock, Conv1x1Projection, (2, 2, 2,), 10)

    x = torch.ones(size=(10, 3, 32, 32)) # (N, C, H, W)
    y = resnet(x)

    assert y.shape == (10,10)

def test_resnet_zero_padding_max_pool():
    resnet = ResNet(ResidualBlock, ZeroPaddingMaxPool, (2, 2, 2,), 10)

    x = torch.ones(size=(10, 3, 32, 32)) # (N, C, H, W)
    y = resnet(x)

    assert y.shape == (10,10)

def test_plain_net():
    plain = ResNet(PlainBlock, None, [3, 3, 3])

    x = torch.ones(size=(10, 3, 32, 32)) # (N, C, H, W)
    y = plain(x)

    assert y.shape == (10,10)

def test_zero_padding():

    # no padding applied / spatially down-sampled
    x = torch.ones(size=(1, 16, 32, 32)) # (N, C, H, W)
    zero_padding = ZeroPadding(16, 16)
    y = zero_padding(x)

    assert y.shape == (1, 16, 16, 16)

    # padding of 16 applied / spatially down-sampled
    x = torch.ones(size=(1, 16, 32, 32)) # (N, C, H, W)
    zero_padding = ZeroPadding(16, 32)
    y = zero_padding(x)

    assert y.shape == (1, 32, 16, 16)

def test_conv1x1_projection():
    x = torch.ones(size=(1, 16, 32, 32))

    projection = Conv1x1Projection(16, 32)
    y = projection(x)

    assert y.shape == (1, 32, 16, 16)

def test_zero_padding_max_pool  ():
    x = torch.ones(size=(1, 16, 32, 32))

    projection = ZeroPaddingMaxPool(16, 32)
    y = projection(x)

    assert y.shape == (1, 32, 16, 16)