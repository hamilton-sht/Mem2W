import torch
from swift.mem2w.dual_trainer import chunked_hidden_cross_entropy, masked_causal_cross_entropy


def test_chunked_projection_matches_full_sequence_loss_and_gradient():
    torch.manual_seed(42)
    head = torch.nn.Linear(7, 23).requires_grad_(False)
    hidden = torch.randn(2, 17, 7, requires_grad=True)
    labels = torch.randint(0, 23, (2, 17))
    labels[:, :5] = -100
    labels[0, 8:12] = -100
    expected, n = masked_causal_cross_entropy(head(hidden), labels)
    expected.backward()
    gradient = hidden.grad.clone()
    for chunk_size in (1, 3, 100):
        candidate = hidden.detach().clone().requires_grad_()
        actual, m = chunked_hidden_cross_entropy(candidate, head, labels, chunk_size)
        actual.backward()
        assert n == m
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(candidate.grad, gradient)
    assert all(p.grad is None for p in head.parameters())


def test_shifted_loss_counts_every_local_target():
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[3, 1, 2, -100]])
    _loss, count = masked_causal_cross_entropy(logits, labels, labels_are_shifted=True)
    assert count == 3
    _loss, count = chunked_hidden_cross_entropy(
        logits.detach(), torch.nn.Linear(5, 5, bias=False), labels, 2, labels_are_shifted=True)
    assert count == 3


if __name__ == '__main__':
    test_chunked_projection_matches_full_sequence_loss_and_gradient()
    test_shifted_loss_counts_every_local_target()
    print('chunked CE: value, count, gradients, frozen head PASSED', flush=True)
