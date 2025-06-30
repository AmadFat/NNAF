from anytree import LevelOrderIter, Node
from nnaf_utils.pytype import *

from ..pttype import *


class Deriviator:
    def __init__(
        self,
        vars: dict[str, Any],
    ):
        r"""Automatic high-dimensional deriviation.

        Example:
            >>> x = torch.tensor([1.0], requires_grad=True)
            >>> t = torch.tensor([2.0], requires_grad=True)
            >>> u = x**3 * t**2
            >>> deriviator = Deriviator({'x': x, 't': t})
            >>> results = deriviator(
            ...     u,
            ...     u_x=['x'],              # = 3x²t² = 12
            ...     u_t=['t'],              # = 2x³t = 4
            ...     u_xx=['x', 'x'],        # = 6xt² = 24
            ...     u_xt=['x', 't'],        # = 6x²t = 12
            ...     u_tt=['t', 't'],        # = 2x³ = 2
            ...     u_xxx=['x', 'x', 'x'],  # = 6t² = 24
            ...     u_xtt=['x', 't', 't'],  # = 6x² = 6
            ... )
            >>> for name, tensor in results.items():
            ...     print(f"{name} = {tensor.item()}").

        """
        self.vars = vars

    def append(
        self,
        identity: str,
        strflow: list[str],
    ):
        has_parent = False
        for node in LevelOrderIter(self.root, filter_=lambda n: n.depth == len(strflow) - 1):
            if node.strflow == strflow[:-1]:
                has_parent = True
                node = Node(identity, strflow=strflow, parent=node, var=strflow[-1])
                break
        if not has_parent:
            parent = self.append(f"{identity}'s parent", strflow=strflow[:-1])
            node = Node(identity, strflow=strflow, parent=parent, var=strflow[-1])
        return node

    def __call__(
        self,
        output: Any,
        **flows: dict[str, list[str]],
    ):
        self.root = Node("root", strflow=[], grad=output)

        flows = [(identity, strflow) for identity, strflow in flows.items()]
        identities, strflows = zip(*sorted(flows, key=lambda x: len(x[1]), reverse=False), strict=False)
        for identity, strflow in zip(identities, strflows, strict=False):
            assert all(s in self.vars for s in strflow)
            self.append(identity, strflow)

        results = {}
        for node in LevelOrderIter(self.root, filter_=lambda n: n.depth > 0):
            node.grad = torch.autograd.grad(
                node.parent.grad, self.vars[node.var], retain_graph=True, create_graph=True
            )[0]
            if node.name in identities:
                results[node.name] = node.grad

        for node in LevelOrderIter(self.root):
            node.parent = None

        return results
