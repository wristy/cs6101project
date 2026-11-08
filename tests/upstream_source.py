"""Helpers to check that a variant's method is the upstream method except for named lines."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACER = ROOT / "external/thought-tracing/tracer.py"


def method_body(path, class_name, method_name):
    """The method's statements as ast dumps, without its docstring."""
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    body = item.body
                    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                        body = body[1:]
                    return [ast.dump(statement) for statement in body]
    raise LookupError(f"{class_name}.{method_name} not found in {path}")


def differing_statements(upstream, variant):
    """Indices where two equal-length statement lists differ."""
    assert len(upstream) == len(variant), "variant adds or removes statements"
    return [index for index, (a, b) in enumerate(zip(upstream, variant)) if a != b]
