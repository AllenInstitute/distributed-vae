import os
import random
import string
from collections.abc import Sequence
from functools import reduce


from pyrsistent import m, v, pmap, pvector, PMap, PVector

def bold(s):
  match s:
    case int(x):
      return f"\033[1mrank {s}\033[0m"
    case _:
      return f"\033[1m{s}\033[0m"

def conj(coll, item, value=None):
  if isinstance(coll, PVector):
    return coll.append(item)
  elif isinstance(coll, PMap):
    return coll.set(item, value)
  
def try_to_get(obj, key, default=None):
  v = obj.get(key, default)
  return obj.set(key, v)

def wrap_with(x, wrap_type):
  match wrap_type:
    case 'seq':
      if isinstance(x, Sequence):
        return pvector([x])
  return x

def pprint(dct):
  def format_value(v):
    if isinstance(v, dict):
      return '{' + ', '.join(f'{repr(k)}: {format_value(v)}' for k, v in v.items()) + '}'
    elif isinstance(v, list):
      return '[' + ', '.join(format_value(item) for item in v) + ']'
    else:
      return repr(v)

  result = ''
  for i, (k, v) in enumerate(sorted(dct.items(), key=lambda kv: kv[0])):
    if i == 0:
      result += "{"
    else:
      result += " "
    result += f"{repr(k)}: {format_value(v)}"
    if i < len(dct) - 1:
      result += ',\n'
  result += '}'
  print(result)

def take_percent(lst, percent):
  if isinstance(lst, Sequence):
    return lst[:int(len(lst) * percent)]

def starfilter(pred, iterable):
  for args in iterable:
    if pred(*args):
      yield args

def starfilter(pred, iterable):
  for args in iterable:
    if pred(*args):
      yield args

# TODO
def star(fun):
  return lambda args: fun(*args)

def avg(xs):
  return sum(xs) / len(xs)

def take(lst, n, uniform=False, **kwargs):
  if uniform:
    return reduce(lambda output, i: output.append(lst[(len(lst) // n) * i]), 
                  range(min(n, len(lst))), 
                  v())
  return lst[:n]

def params(model):
  return pvector(model.parameters())

# TODO: refactor
def cached_dprint():
  color_code = pmap({
    'black': '30',
    'red': '31',
    'green': '32',
    'yellow': '33',
    'blue': '34',
    'magenta': '35',
    'cyan': '36',
    'white': '37',
  })
  def dprint(*args, color=None, bold=False, italics=False, **kwargs):
    if __debug__:
      start_code = "\033["
      end_code = "\033[0m"
      codes = v()
      if bold:
        codes = codes.append('1')
      if italics:
        codes = codes.append('3')
      if color is not None:
        codes = codes.append(color_code[color])
      if len(codes) > 0:
        start_code += ';'.join(codes) + 'm'
        print(start_code, end='')
      print(*args, **kwargs)
      print(end_code, end='')
  return dprint

_dprint = cached_dprint()

def dprint(*args, **kwargs):
  return _dprint(*args, **kwargs)

def make_path(path):
  if not os.path.exists(path):
    os.makedirs(path)

def convert(x, src, dst):
  if src == dst:
    return x
  match src, dst:
    case 'B', 'MB':
      return x / 1024**2
    
def random_string(n):
  return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def random_int(n):
  return random.randint(10**(n-1), (10**n)-1)

def random_of(tp, n):
  if tp == 'str':
    return random_string(n)
  elif tp == 'int':
    return random_int(n)
  
def debug(fun):
  def f(*args, **kwargs):
    if __debug__:
      return fun(*args, **kwargs)
  return f if __debug__ else lambda *args, **kwargs: None


