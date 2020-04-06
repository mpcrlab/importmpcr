from .infix import *
from collections.abc import Iterable, Callable

@make_infix('mod','pow','rshift','mul','div','fdiv')
def to(start,end):
    return list(range(start,end))

@make_infix('mod','pow','rshift','mul','div','fdiv')
def take(gen,count):
    if isinstance(gen,Iterable):
        return [next(gen) for _ in range(count)]
    elif isinstance(gen,Callable):
        return [gen() for _ in range(count)]
    else:
        raise RuntimeError("What did you give me")

@make_infix('mod','pow','rshift','mul','div','fdiv')
def of(func,list):
    return [func(i) for i in list]
