from multimethod import multimethod, multidispatch
from functools import singledispatch
from typing import Any

def foo(*args, **kwargs):
  match args, kwargs:
    case (a,), {}:
      return 1
    case (a, b), {}:
      return 2

def main():
  m1 = {'a': 1, 'b': 2}
  print(m1.a)
  print(foo(1))
  # print(foo(1, 2))

if __name__ == '__main__':
  main()