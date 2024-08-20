# type: ignore
# ruff: noqa

"""
Copied from https://github.com/saprmarks/feature-circuits/blob/main/attribution.py for comparison with our implementation in tests
"""

from collections import namedtuple

import torch as t

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = {"validate": True, "scan": True}
else:
    tracer_kwargs = {"validate": False, "scan": False}

EffectOut = namedtuple("EffectOut", ["effects", "deltas", "grads", "total_effect"])


class SparseAct:
    """
    A SparseAct is a helper class which represents a vector in the sparse feature basis provided by an SAE, jointly with the SAE error term.
    A SparseAct may have three fields:
    act : the feature activations in the sparse basis
    res : the SAE error term
    resc : a contracted SAE error term, useful for when we want one number per feature and error (instead of having d_model numbers per error)
    """

    def __init__(
        self,
        act: t.Tensor = None,
        res: t.Tensor = None,
        resc: t.Tensor = None,  # contracted residual
    ) -> None:
        self.act = act
        self.res = res
        self.resc = resc

    def _map(self, f, aux=None) -> "SparseAct":
        kwargs = {}
        if isinstance(aux, SparseAct):
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None and getattr(aux, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), getattr(aux, attr))
        else:
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), aux)
        return SparseAct(**kwargs)

    def __mul__(self, other) -> "SparseAct":
        if isinstance(other, SparseAct):
            # Handle SparseAct * SparseAct
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * getattr(other, attr)
        else:
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * other
        return SparseAct(**kwargs)

    def __rmul__(self, other) -> "SparseAct":
        # This will handle float/int * SparseAct by reusing the __mul__ logic
        return self.__mul__(other)

    def __matmul__(self, other: "SparseAct") -> "SparseAct":
        # dot product between two SparseActs, except only the residual is contracted
        return SparseAct(
            act=self.act * other.act,
            resc=(self.res * other.res).sum(dim=-1, keepdim=True),
        )

    def __add__(self, other) -> "SparseAct":
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(
                            f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}"
                        )
                    kwargs[attr] = getattr(self, attr) + getattr(other, attr)
        else:
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) + other
        return SparseAct(**kwargs)

    def __radd__(self, other: "SparseAct") -> "SparseAct":
        return self.__add__(other)

    def __sub__(self, other: "SparseAct") -> "SparseAct":
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(
                            f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}"
                        )
                    kwargs[attr] = getattr(self, attr) - getattr(other, attr)
        else:
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) - other
        return SparseAct(**kwargs)

    def __truediv__(self, other) -> "SparseAct":
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / getattr(other, attr)
        else:
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / other
        return SparseAct(**kwargs)

    def __rtruediv__(self, other) -> "SparseAct":
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        else:
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        return SparseAct(**kwargs)

    def __neg__(self) -> "SparseAct":
        sparse_result = -self.act
        res_result = -self.res
        return SparseAct(act=sparse_result, res=res_result)

    def __invert__(self) -> "SparseAct":
        return self._map(lambda x, _: ~x)

    def __gt__(self, other) -> "SparseAct":
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) > other
            return SparseAct(**kwargs)
        raise ValueError("SparseAct can only be compared to a scalar.")

    def __lt__(self, other) -> "SparseAct":
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ["act", "res", "resc"]:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) < other
            return SparseAct(**kwargs)
        raise ValueError("SparseAct can only be compared to a scalar.")

    def __getitem__(self, index: int):
        return self.act[index]

    def __repr__(self):
        if self.res is None:
            return f"SparseAct(act={self.act}, resc={self.resc})"
        if self.resc is None:
            return f"SparseAct(act={self.act}, res={self.res})"
        else:
            raise ValueError(
                "SparseAct has both residual and contracted residual. This is an unsupported state."
            )

    def sum(self, dim=None):
        kwargs = {}
        for attr in ["act", "res", "resc"]:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).sum(dim)
        return SparseAct(**kwargs)

    def mean(self, dim: int):
        kwargs = {}
        for attr in ["act", "res", "resc"]:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).mean(dim)
        return SparseAct(**kwargs)

    def nonzero(self):
        kwargs = {}
        for attr in ["act", "res", "resc"]:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).nonzero()
        return SparseAct(**kwargs)

    def squeeze(self, dim: int):
        kwargs = {}
        for attr in ["act", "res", "resc"]:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).squeeze(dim)
        return SparseAct(**kwargs)

    @property
    def grad(self):
        kwargs = {}
        for attribute in ["act", "res", "resc"]:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).grad
        return SparseAct(**kwargs)

    def clone(self):
        kwargs = {}
        for attribute in ["act", "res", "resc"]:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).clone()
        return SparseAct(**kwargs)

    @property
    def value(self):
        kwargs = {}
        for attribute in ["act", "res", "resc"]:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).value
        return SparseAct(**kwargs)

    def save(self):
        for attribute in ["act", "res", "resc"]:
            if getattr(self, attribute) is not None:
                setattr(self, attribute, getattr(self, attribute).save())
        return self

    def detach(self):
        self.act = self.act.detach()
        self.res = self.res.detach()
        return SparseAct(act=self.act, res=self.res)

    def to_tensor(self):
        if self.resc is None:
            return t.cat([self.act, self.res], dim=-1)
        if self.res is None:
            return t.cat([self.act, self.resc], dim=-1)
        raise ValueError(
            "SparseAct has both residual and contracted residual. This is an unsupported state."
        )

    def to(self, device):
        for attr in ["act", "res", "resc"]:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        return self

    def __gt__(self, other):
        return self._map(lambda x, y: x > y, other)

    def __lt__(self, other):
        return self._map(lambda x, y: x < y, other)

    def nonzero(self):
        return self._map(lambda x, _: x.nonzero())

    def squeeze(self, dim):
        return self._map(lambda x, _: x.squeeze(dim=dim))

    def expand_as(self, other):
        return self._map(lambda x, y: x.expand_as(y), other)

    def zeros_like(self):
        return self._map(lambda x, _: t.zeros_like(x))

    def ones_like(self):
        return self._map(lambda x, _: t.ones_like(x))

    def abs(self):
        return self._map(lambda x, _: x.abs())


def pe_ig(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
):
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(
                act=f.save(), res=residual.save()
            )
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k: SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res))
            for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.no_grad():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(
                    act=f.save(), res=residual.save()
                )
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k: v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean, scan=tracer_kwargs["scan"]):
                    if is_tuple[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(
                retain_graph=True
            )  # TODO : why is this necessary? Probably shouldn't be, contact jaden

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (
            (patch_state - clean_state).detach()
            if patch_state is not None
            else -clean_state.detach()
        )
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return EffectOut(effects, deltas, grads, total_effect)
