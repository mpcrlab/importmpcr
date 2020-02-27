from .infix import *

@make_infix('mod','pow','rshift','mul','div','fdiv')
def to(start,end):
    return list(range(start,end))

@make_infix('mod','pow','rshift','mul','div','fdiv')
def take(gen,count):
    return [next(gen) for _ in range(count)]

@make_infix('mod','pow','rshift','mul','div','fdiv')
def of(func,list):
    return [func(i) for i in list]
