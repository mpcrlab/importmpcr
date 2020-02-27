from .infix import *

@make_infix('mod','pow','rshift','mul')
def to(start,end):
    return list(range(start,end))

@make_infix('mod','pow','rshift','mul')
def take(gen,count):
    return [next(gen) for _ in range(count)]

@make_infix('mod','pow','rshift','mul')
def to(start,end):
    return list(range(start,end+1))
