import torch


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def artanh(x):
    x = x.clamp(-1 + 1e-5, 1 - 1e-5)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def expmap0(u, c=1.0):
    sqrt_c = torch.as_tensor(c, dtype=u.dtype, device=u.device).sqrt()
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(1e-5)
    return tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)


def project(x, c=1.0):
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-5)
    maxnorm = (1 - 1e-3) / c.sqrt()
    projected = x / norm * maxnorm
    return torch.where(norm > maxnorm, projected, x)


def mobius_add(x, y, c=1.0):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return numerator / (denominator + 1e-5)


def dist(x, y, c=1.0, keepdim=False):
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    sqrt_c = c.sqrt()
    delta = mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim)
    return 2 / sqrt_c * artanh(sqrt_c * delta)


def _mobius_addition_batch(x, y, c):
    xy = x @ y.transpose(-2, -1)
    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    numerator = 1 + 2 * c * xy + c * y2.transpose(-2, -1)
    numerator = numerator.unsqueeze(-1) * x.unsqueeze(-2)
    numerator = numerator + (1 - c * x2).unsqueeze(-2) * y.unsqueeze(-3)
    denominator = 1 + 2 * c * xy + c ** 2 * x2 * y2.transpose(-2, -1)
    return numerator / (denominator.unsqueeze(-1) + 1e-5)


def dist_matrix(x, y, c=1.0):
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    sqrt_c = c.sqrt()
    delta = _mobius_addition_batch(-x, y, c).norm(dim=-1)
    return 2 / sqrt_c * artanh(sqrt_c * delta)


def mobius_matvec(matrix, x, c=1.0):
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    sqrt_c = c.sqrt()
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-5)
    mx = x @ matrix.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
    safe_norm = mx_norm.clamp_min(1e-5)
    scale = tanh(safe_norm / x_norm * artanh(sqrt_c * x_norm)) / (sqrt_c * safe_norm)
    result = scale * mx
    result = torch.where(mx_norm > 1e-5, result, torch.zeros_like(result))
    return project(result, c=c)
