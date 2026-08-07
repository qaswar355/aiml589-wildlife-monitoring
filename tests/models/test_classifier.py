"""Tests for load_checkpoint -- the shared checkpoint-loading helper used
by evaluate.py, gradcam.py, and the inference API."""
import torch

from src.models.classifier import WildlifeClassifier, load_checkpoint


def _save_tiny_checkpoint(tmp_path, dropout_rate=0.3):
    model = WildlifeClassifier(architecture="efficientnet_b3", pretrained=False, dropout_rate=dropout_rate)
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return model, checkpoint_path


def test_load_checkpoint_restores_weights_exactly(tmp_path):
    saved_model, checkpoint_path = _save_tiny_checkpoint(tmp_path)
    loaded_model = load_checkpoint(str(checkpoint_path), architecture="efficientnet_b3")

    for (name, saved_param), (_, loaded_param) in zip(
        saved_model.state_dict().items(), loaded_model.state_dict().items()
    ):
        assert torch.equal(saved_param, loaded_param), f"mismatch in {name}"


def test_load_checkpoint_defaults_to_cpu_device(tmp_path):
    _, checkpoint_path = _save_tiny_checkpoint(tmp_path)
    model = load_checkpoint(str(checkpoint_path), architecture="efficientnet_b3")
    assert next(model.parameters()).device == torch.device("cpu")


def test_load_checkpoint_applies_given_dropout_rate(tmp_path):
    _, checkpoint_path = _save_tiny_checkpoint(tmp_path, dropout_rate=0.3)
    model = load_checkpoint(str(checkpoint_path), architecture="efficientnet_b3", dropout_rate=0.5)

    dropout_layer = model.backbone.classifier[0]
    assert isinstance(dropout_layer, torch.nn.Dropout)
    assert dropout_layer.p == 0.5


def test_load_checkpoint_returns_eval_mode(tmp_path):
    _, checkpoint_path = _save_tiny_checkpoint(tmp_path)
    model = load_checkpoint(str(checkpoint_path), architecture="efficientnet_b3")
    assert model.training is False


def test_predict_proba_mc_dropout_does_not_mutate_batchnorm_running_stats(tmp_path):
    """Regression test: predict_proba(mc_dropout_passes>1) used to call
    self.train(True), which also switches every BatchNorm layer in the
    backbone into training mode -- computing statistics from just the one
    input image instead of using the learned running averages, and
    permanently drifting those running averages on every single call.
    Only Dropout should ever switch on for MC sampling."""
    _, checkpoint_path = _save_tiny_checkpoint(tmp_path)
    model = load_checkpoint(str(checkpoint_path), architecture="efficientnet_b3")

    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bn_layers, "expected at least one BatchNorm2d in the EfficientNet-B3 backbone"
    before = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in bn_layers]

    x = torch.randn(1, 3, 300, 300)
    model.predict_proba(x, mc_dropout_passes=20)

    for bn, (mean_before, var_before) in zip(bn_layers, before):
        assert torch.equal(bn.running_mean, mean_before)
        assert torch.equal(bn.running_var, var_before)


def test_predict_proba_mc_dropout_leaves_model_in_eval_mode_after(tmp_path):
    _, checkpoint_path = _save_tiny_checkpoint(tmp_path)
    model = load_checkpoint(str(checkpoint_path), architecture="efficientnet_b3")

    x = torch.randn(1, 3, 300, 300)
    model.predict_proba(x, mc_dropout_passes=20)

    assert model.training is False
    for m in model.modules():
        assert m.training is False
