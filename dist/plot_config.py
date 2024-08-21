from pyrsistent import pmap
from functools import partial
from itertools import product


STRATS = ['full', 'ddp', 'hybrid', 'grad-op', 'no', 'hybrid-zero2']
MODELS = ['shallow', 'deep']
GPUS = ['1', '2', '3', '4']
BASE = pmap({
    'model': 'shallow',
    'batch_size': '256',
    'num_workers': '4',
    'epochs': '100',
    'gpus': '1',
    'use_dist_sampler': 'False',
    'sharding_strategy': 'full',
    'mixed_precision': 'none',
    'sync_module_states': 'False',
    'use_compilation': 'False',
    'use_orig_params': 'False',
    'use_batchnorm': 'True'
})

def foreach(f, xs):
    for x in xs: f(*x)

def numname(n):
    ws = {
        '0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
        '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine',
        '10': 'Ten', '11': 'Eleven', '12': 'Twelve', '13': 'Thirteen',
        '14': 'Fourteen', '15': 'Fifteen', '16': 'Sixteen',
        '17': 'Seventeen', '18': 'Eighteen', '19': 'Nineteen',
        '20': 'Twenty', '30': 'Thirty', '40': 'Forty', '50': 'Fifty',
        '60': 'Sixty', '70': 'Seventy', '80': 'Eighty', '90': 'Ninety'
    }
    assert not len(n) > 2; return ws[n] if n in ws else ws[n[0] + '0'] + ws[n[1]]
    
def upper(s):
    return s.upper()

def tagged_combos(**xss):
    return map(lambda vs: pmap(zip(xss.keys(), vs)), product(*xss.values()))

def dist_sampler_map(xs):
    return map(lambda x: x.set('use_dist_sampler', 'True') if x.gpus != '1' else x, xs)

def dedup(xs):
    return (lambda st: [x for x in xs if not (x in st or st.add(x))] )(set())

def _gen_configs(base, **kw):
    return dedup(map(base.update, dist_sampler_map(tagged_combos(**kw))))

def gen_name(kw):
    return upper(f'{numname(kw["gpus"])}_GPUs_{kw["model"]}_{kw["sharding_strategy"]}')

def gen_configs(base, **kw):
    return {gen_name(c): c for c in _gen_configs(base, **kw)}

def init_module_(base, **kw):
    foreach(lambda k, v: globals().__setitem__(k, v), gen_configs(base, **kw).items())

init_module_(BASE, gpus=GPUS, model=MODELS, sharding_strategy=STRATS)





# -- full sharding --
"""
One GPU Shallow: r1, r44, r45
Two GPUs Shallow: r37, r38, r39, r164, r165, r166
Three GPUs Shallow: r6, r12, r42, r118, r119, r120
Four GPUs Shallow: r2, r3, r4, r66, r68, r70

_plot('losses', dirn='toy-runs', title='Train loss per epoch (shallow model)', ylabel='Train loss',
r1='One GPU', r44='One GPU', r45='One GPU',
r37='Two GPUs', r38='Two GPUs', r39='Two GPUs',
r6='Three GPUs', r12='Three GPUs', r42='Three GPUs',
r2='Four GPUs', r3='Four GPUs', r4='Four GPUs')
_plot('mem_summary', dirn='toy-runs', style='bar', title='Memory allocated in MB (shallow model)', ylabel='Memory (MB)',
r1='One GPU', r44='One GPU', r45='One GPU',
r37='Two GPUs', r38='Two GPUs', r39='Two GPUs',
r6='Three GPUs', r12='Three GPUs', r42='Three GPUs',
r2='Four GPUs', r3='Four GPUs', r4='Four GPUs')
_plot('epoch_times', style='bar', use_average=True, dirn='toy-runs', title='Seconds per epoch (shallow model)', ylabel='Epoch time (s)',
r1='One GPU', r44='One GPU', r45='One GPU',
r37='Two GPUs', r38='Two GPUs', r39='Two GPUs',
r6='Three GPUs', r12='Three GPUs', r42='Three GPUs',
r2='Four GPUs', r3='Four GPUs', r4='Four GPUs')


One GPU Deep: r179, r182, r184
Two GPUs Deep: r176, r177, r183
Three GPUs Deep: r46, r47, r50, r121, r122
Four GPUs Deep: r5, r7, r8, r73, r75, r77
_plot('losses', dirn='toy-runs', title='Train loss per epoch (deep model)', ylabel='Train loss',
r179='One GPU', r182='One GPU', r184='One GPU',
r176='Two GPUs', r177='Two GPUs', r183='Two GPUs',
r46='Three GPUs', r47='Three GPUs', r50='Three GPUs',
r73='Four GPUs', r75='Four GPUs', r77='Four GPUs')
_plot('mem_summary', dirn='toy-runs', style='bar', title='Memory allocated in MB (deep model)', ylabel='Memory (MB)',
r179='One GPU', r182='One GPU', r184='One GPU',
r176='Two GPUs', r177='Two GPUs', r183='Two GPUs',
r46='Three GPUs', r47='Three GPUs', r50='Three GPUs',
r73='Four GPUs', r75='Four GPUs', r77='Four GPUs')
_plot('epoch_times', style='bar', use_average=True, dirn='toy-runs', title='Seconds per epoch (deep model)', ylabel='Epoch time (s)',
r179='One GPU', r182='One GPU', r184='One GPU',
r176='Two GPUs', r177='Two GPUs', r183='Two GPUs',
r46='Three GPUs', r47='Three GPUs', r50='Three GPUs',
r73='Four GPUs', r75='Four GPUs', r77='Four GPUs')
"""

# DDP
"""
One GPU Shallow: r1, r44, r45
Two GPUs Shallow: r134, r135, r137
Three GPUs Shallow: r79, r80, r81
Four GPUs Shallow:  r16, r17, r18
_plot('losses', dirn='toy-runs', title=' DDP Train loss per epoch (shallow model)', ylabel='Train loss',
r1='One GPU', r44='One GPU', r45='One GPU',
r134='Two GPUs', r135='Two GPUs', r137='Two GPUs',
r79='Three GPUs', r80='Three GPUs', r81='Three GPUs',
r16='Four GPUs', r17='Four GPUs', r18='Four GPUs')
_plot('val_losses', dirn='toy-runs', title=' DDP Validation loss per epoch (shallow model)', ylabel='Validation loss',
r1='One GPU', r44='One GPU', r45='One GPU',
r134='Two GPUs', r135='Two GPUs', r137='Two GPUs',
r79='Three GPUs', r80='Three GPUs', r81='Three GPUs',
r16='Four GPUs', r17='Four GPUs', r18='Four GPUs')
_plot('mem_summary', dirn='toy-runs', style='bar', title='DDP Memory allocated in MB (shallow model)', ylabel='Memory (MB)',
r1='One GPU', r44='One GPU', r45='One GPU',
r134='Two GPUs', r135='Two GPUs', r137='Two GPUs',
r79='Three GPUs', r80='Three GPUs', r81='Three GPUs',
r16='Four GPUs', r17='Four GPUs', r18='Four GPUs')
_plot('epoch_times', style='bar', use_average=True, dirn='toy-runs', title='DDP Seconds per epoch (shallow model)', ylabel='Epoch time (s)',
r1='One GPU', r44='One GPU', r45='One GPU',
r134='Two GPUs', r135='Two GPUs', r137='Two GPUs',
r79='Three GPUs', r80='Three GPUs', r81='Three GPUs',
r16='Four GPUs', r17='Four GPUs', r18='Four GPUs')

One GPU Deep: r179, r182, r184
Two GPUs Deep: r138, r185, r186
Three GPUs Deep: r82, r83, r84
Four GPUs Deep: r19, r20, r21
_plot('losses', dirn='toy-runs', title=' DDP Train loss per epoch (deep model)', ylabel='Train loss',
r179='One GPU', r182='One GPU', r184='One GPU',
r138='Two GPUs', r185='Two GPUs', r186='Two GPUs',
r82='Three GPUs', r83='Three GPUs', r84='Three GPUs',
r19='Four GPUs', r20='Four GPUs', r21='Four GPUs')
_plot('val_losses', dirn='toy-runs', title=' DDP Validation loss per epoch (deep model)', ylabel='Validation loss',
r179='One GPU', r182='One GPU', r184='One GPU',
r138='Two GPUs', r185='Two GPUs', r186='Two GPUs',
r82='Three GPUs', r83='Three GPUs', r84='Three GPUs',
r19='Four GPUs', r20='Four GPUs', r21='Four GPUs')
_plot('mem_summary', dirn='toy-runs', style='bar', title='DDP Memory allocated in MB (deep model)', ylabel='Memory (MB)',
r179='One GPU', r182='One GPU', r184='One GPU',
r138='Two GPUs', r185='Two GPUs', r186='Two GPUs',
r82='Three GPUs', r83='Three GPUs', r84='Three GPUs',
r19='Four GPUs', r20='Four GPUs', r21='Four GPUs')
_plot('epoch_times', style='bar', use_average=True, dirn='toy-runs', title='DDP Seconds per epoch (deep model)', ylabel='Epoch time (s)',
r179='One GPU', r182='One GPU', r184='One GPU',
r138='Two GPUs', r185='Two GPUs', r186='Two GPUs',
r82='Three GPUs', r83='Three GPUs', r84='Three GPUs',
r19='Four GPUs', r20='Four GPUs', r21='Four GPUs')
"""

# hybrid
"""
One GPU Shallow: r1, r44, r45
Two GPUs Shallow: r145, r146, r147
Three GPUs Shallow: r91, r92, r93
Four GPUs Shallow: r28, r29, r30
_plot('losses', dirn='toy-runs', title=' Hybrid Train loss per epoch (shallow model)', ylabel='Train loss',
r1='One GPU', r44='One GPU', r45='One GPU',
r145='Two GPUs', r146='Two GPUs', r147='Two GPUs',
r91='Three GPUs', r92='Three GPUs', r93='Three GPUs',
r28='Four GPUs', r29='Four GPUs', r30='Four GPUs')
_plot('val_losses', dirn='toy-runs', title=' Hybrid Validation loss per epoch (shallow model)', ylabel='Validation loss',
r1='One GPU', r44='One GPU', r45='One GPU',
r145='Two GPUs', r146='Two GPUs', r147='Two GPUs',
r91='Three GPUs', r92='Three GPUs', r93='Three GPUs',
r28='Four GPUs', r29='Four GPUs', r30='Four GPUs')
_plot('mem_summary', dirn='toy-runs', style='bar', title='Hybrid Memory allocated in MB (shallow model)', ylabel='Memory (MB)',
r1='One GPU', r44='One GPU', r45='One GPU',
r145='Two GPUs', r146='Two GPUs', r147='Two GPUs',
r91='Three GPUs', r92='Three GPUs', r93='Three GPUs',
r28='Four GPUs', r29='Four GPUs', r30='Four GPUs')
_plot('epoch_times', style='bar', use_average=True, dirn='toy-runs', title='Hybrid Seconds per epoch (shallow model)', ylabel='Epoch time (s)',
r1='One GPU', r44='One GPU', r45='One GPU',
r145='Two GPUs', r146='Two GPUs', r147='Two GPUs',
r91='Three GPUs', r92='Three GPUs', r93='Three GPUs',
r28='Four GPUs', r29='Four GPUs', r30='Four GPUs')

One GPU Deep: r179, r182, r184
Two GPUs Deep: r187, r188, r189
Three GPUs Deep: r178, r180, r181
Four GPUs Deep: r31, r32, r33
_plot('losses', dirn='toy-runs', title=' Hybrid Train loss per epoch (deep model)', ylabel='Train loss',
r179='One GPU', r182='One GPU', r184='One GPU',
r187='Two GPUs', r188='Two GPUs', r189='Two GPUs',
r178='Three GPUs', r180='Three GPUs', r181='Three GPUs',
r31='Four GPUs', r32='Four GPUs', r33='Four GPUs')
_plot('val_losses', dirn='toy-runs', title=' Hybrid Validation loss per epoch (deep model)', ylabel='Validation loss',
r179='One GPU', r182='One GPU', r184='One GPU',
r187='Two GPUs', r188='Two GPUs', r189='Two GPUs',
r178='Three GPUs', r180='Three GPUs', r181='Three GPUs',
r31='Four GPUs', r32='Four GPUs', r33='Four GPUs')
_plot('mem_summary', dirn='toy-runs', style='bar', title='Hybrid Memory allocated in MB (deep model)', ylabel='Memory (MB)',
r179='One GPU', r182='One GPU', r184='One GPU',
r187='Two GPUs', r188='Two GPUs', r189='Two GPUs',
r178='Three GPUs', r180='Three GPUs', r181='Three GPUs',
r31='Four GPUs', r32='Four GPUs', r33='Four GPUs')
_plot('epoch_times', style='bar', use_average=True, dirn='toy-runs', title='Hybrid Seconds per epoch (deep model)', ylabel='Epoch time (s)',
r179='One GPU', r182='One GPU', r184='One GPU',
r187='Two GPUs', r188='Two GPUs', r189='Two GPUs',
r178='Three GPUs', r180='Three GPUs', r181='Three GPUs',
r31='Four GPUs', r32='Four GPUs', r33='Four GPUs')
"""

# no sampler
"""
One GPU Shallow: r1, r44, r45
Two GPUs Shallow: r103, r104, r105
Three GPUs Shallow: r52, r54, r56
Four GPUs Shallow: r202, r203, r204
_plot('losses', dirn='toy-runs', title=' No sampler Train loss per epoch (shallow model)', ylabel='Train loss',
r1='One GPU', r44='One GPU', r45='One GPU',
r103='Two GPUs', r104='Two GPUs', r105='Two GPUs',
r52='Three GPUs', r54='Three GPUs', r56='Three GPUs',
r198='Four GPUs', r200='Four GPUs', r201='Four GPUs')
_plot('val_losses', dirn='toy-runs', title=' No sampler Validation loss per epoch (shallow model)', ylabel='Validation loss',
r1='One GPU', r44='One GPU', r45='One GPU',
r103='Two GPUs', r104='Two GPUs', r105='Two GPUs',
r52='Three GPUs', r54='Three GPUs', r56='Three GPUs',
r198='Four GPUs', r200='Four GPUs', r201='Four GPUs')
_plot('mem_summary', dirn='toy-runs', style='bar', title='No sampler Memory allocated in MB (shallow model)', ylabel='Memory (MB)',
r1='One GPU', r44='One GPU', r45='One GPU',
r103='Two GPUs', r104='Two GPUs', r105='Two GPUs',
r52='Three GPUs', r54='Three GPUs', r56='Three GPUs',
r198='Four GPUs', r200='Four GPUs', r201='Four GPUs')
_plot('epoch_times', style='bar', use_average=True, dirn='toy-runs', title='No sampler Seconds per epoch (shallow model)', ylabel='Epoch time (s)',
r1='One GPU', r44='One GPU', r45='One GPU',
r103='Two GPUs', r104='Two GPUs', r105='Two GPUs',
r52='Three GPUs', r54='Three GPUs', r56='Three GPUs',
r198='Four GPUs', r200='Four GPUs', r201='Four GPUs')

One GPU Deep: r179, r182, r184
Two GPUs Deep: r193, r196, r199
Three GPUs Deep: r190, r191, r192
Four GPUs Deep: r202, r203, r204
_plot('losses', dirn='toy-runs', title=' No sampler Train loss per epoch (deep model)', ylabel='Train loss',
r179='One GPU', r182='One GPU', r184='One GPU',
r193='Two GPUs', r196='Two GPUs', r199='Two GPUs',
r190='Three GPUs', r191='Three GPUs', r192='Three GPUs',
r202='Four GPUs', r203='Four GPUs', r204='Four GPUs')
_plot('val_losses', dirn='toy-runs', title=' No sampler Validation loss per epoch (deep model)', ylabel='Validation loss',
r179='One GPU', r182='One GPU', r184='One GPU',
r193='Two GPUs', r196='Two GPUs', r199='Two GPUs',
r190='Three GPUs', r191='Three GPUs', r192='Three GPUs',
r202='Four GPUs', r203='Four GPUs', r204='Four GPUs')
_plot('mem_summary', dirn='toy-runs', style='bar', title='No sampler Memory allocated in MB (deep model)', ylabel='Memory (MB)',
r179='One GPU', r182='One GPU', r184='One GPU',
r193='Two GPUs', r196='Two GPUs', r199='Two GPUs',
r190='Three GPUs', r191='Three GPUs', r192='Three GPUs',
r202='Four GPUs', r203='Four GPUs', r204='Four GPUs')
_plot('epoch_times', style='bar', use_average=True, dirn='toy-runs', title='No sampler Seconds per epoch (deep model)', ylabel='Epoch time (s)',
r179='One GPU', r182='One GPU', r184='One GPU',
r193='Two GPUs', r196='Two GPUs', r199='Two GPUs',
r190='Three GPUs', r191='Three GPUs', r192='Three GPUs',
r202='Four GPUs', r203='Four GPUs', r204='Four GPUs')

"""


# list(find_runs(THREE_GPUS_SHALLOW_FULL.set('use_dist_sampler', 'False')))