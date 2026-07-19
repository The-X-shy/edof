from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from edof_reproduction.nafnet import NAFNet  # noqa: E402


def _tiny_nafnet() -> NAFNet:
    return NAFNet(
        width=4,
        middle_blk_num=1,
        enc_blk_nums=[1, 1],
        dec_blk_nums=[1, 1],
    )


def test_public_example_defaults_are_exposed() -> None:
    model = NAFNet()

    assert model.intro.in_channels == 3
    assert model.intro.out_channels == 16
    assert model.ending.out_channels == 3
    assert [len(stage) for stage in model.encoders] == [1, 1, 1, 18]
    assert [len(stage) for stage in model.decoders] == [1, 1, 1, 1]
    assert len(model.middle_blks) == 1
    assert model.padder_size == 16


@pytest.mark.parametrize("height,width", [(13, 17), (16, 20), (1, 7)])
def test_arbitrary_spatial_sizes_are_preserved(height: int, width: int) -> None:
    model = _tiny_nafnet().eval()
    inp = torch.rand(1, 3, height, width)

    with torch.no_grad():
        output = model(inp)

    assert output.shape == inp.shape


def test_cpu_forward_and_backward_are_finite() -> None:
    model = _tiny_nafnet().cpu().train()
    inp = torch.rand(2, 3, 13, 17, requires_grad=True)

    output = model(inp)
    loss = output.square().mean()
    loss.backward()

    assert output.device.type == "cpu"
    assert output.shape == inp.shape
    assert inp.grad is not None
    assert torch.isfinite(inp.grad).all()
    assert model.ending.weight.grad is not None
    assert torch.isfinite(model.ending.weight.grad).all()
    assert model.intro.weight.grad is not None
    assert torch.isfinite(model.intro.weight.grad).all()


def test_state_dict_round_trip_preserves_parameters_and_output() -> None:
    torch.manual_seed(7)
    source = _tiny_nafnet().eval()
    with torch.no_grad():
        source.ending.weight.normal_(mean=0.0, std=0.01)
        source.ending.bias.fill_(0.03)

    state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    restored = _tiny_nafnet().eval()
    restored.load_state_dict(state, strict=True)

    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    inp = torch.rand(1, 3, 11, 19)
    with torch.no_grad():
        expected = source(inp)
        actual = restored(inp)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
