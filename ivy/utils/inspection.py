# global
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints


# local
import ivy


def _is_optional(typ):
    return _is_union(typ) and type(None) in get_args(typ)


def _is_union(typ):
    return get_origin(typ) in {Union, UnionType}


def _is_dict(typ):
    origin = get_origin(typ)
    return origin is dict


def _is_iterable(typ):
    origin = get_origin(typ)
    return origin in {list, tuple}


def _correct_index(is_opt, is_dict, is_iter):
    if is_opt:
        return ["optional"]
    elif is_dict:
        return [str]
    elif is_iter:
        return [int]
    return []


def _get_array_idxs(typ, idx_so_far=None):
    idx_so_far = ivy.default(idx_so_far, [])
    these_idxs = []
    if not hasattr(typ, "__args__"):
        return these_idxs
    is_opt = _is_optional(typ)
    is_union = _is_union(typ)
    is_dict = _is_dict(typ)
    is_iter = _is_iterable(typ)
    for a in typ.__args__:
        a_repr = repr(a)
        if (
            "[" not in a_repr
            and "]" not in a_repr
            and "ivy." in a_repr
            and (".Array" in a_repr or ".NativeArray" in a_repr)
        ):
            these_idxs.append(idx_so_far + _correct_index(is_opt, is_dict, is_iter))
            if is_union:
                break
        else:
            these_idxs += _get_array_idxs(
                a, idx_so_far + _correct_index(is_opt, is_dict, is_iter)
            )
    return these_idxs


def fn_array_spec(fn):
    """
    Return a specification of the function, indicating all arguments which include
    arrays, and the indexes of these.

    Parameters
    ----------
    fn
        function to inspect

    Returns
    -------
    ret
        specification
    """
    try:  # this is because it raises error if python version 3.8.0, in certain cases
        type_hints = get_type_hints(fn)
    except Exception:
        type_hints = {}
    array_idxs = []
    for i, (k, v) in enumerate(type_hints.items()):
        a_idxs = _get_array_idxs(v)
        if not a_idxs:
            continue
        a_idxs = [[(i, k)] + a for a in a_idxs]
        array_idxs += a_idxs
    return array_idxs


def add_array_specs():
    for k, v in ivy.__dict__.items():
        if callable(v) and k[0].islower():
            v.array_spec = fn_array_spec(v)
