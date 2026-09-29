"""Trilha do reel Açovisa: música original (Lá menor, 120 BPM) + efeitos sincronizados às animações.

As deixas de efeito vêm do próprio index.html (window.CUES, exportadas por tools/export_cues.py),
então cada tique de contador, texto revelado, card ou carimbo tem o seu som no tempo exato.
Os efeitos são tonais e limpos (senos, FM, sínteses curtas): sem estrondos longos nem chiado.

Uso: python3 tools/soundtrack.py tools/cues.json saida.wav
"""
import json
import sys

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

SR = 44100
cfg = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
CUES = cfg['cues']
TS = cfg['marks']['TS']
END = cfg['marks']['END']
N = int((END + 0.5) * SR)
rng = np.random.default_rng(7)

BEAT, BAR = 0.5, 2.0
DROP1, BRK, DROP2, OUTRO = TS['s2'], TS['sD'], TS['s5'], TS['s12']
FINAL = 2 * round((OUTRO + 5) / 2)          # acorde final, no início de um compasso

# ---------------------------------------------------------------- harmonia
def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)

CH = {'Am': [57, 60, 64, 69], 'F': [53, 57, 60, 65], 'C': [55, 60, 64, 67], 'G': [55, 59, 62, 67]}
ROOT = {'Am': 45, 'F': 41, 'C': 48, 'G': 43}
MAIN = ['Am', 'F', 'C', 'G']
BRKP = ['F', 'C', 'G', 'Am', 'F', 'G']        # quebra: termina em G para resolver em Am no drop
PEN = [0, 3, 5, 7, 10]                       # pentatônica de Lá menor

def chord_at(t):
    if t >= FINAL:
        return 'Am'
    if FINAL - BAR <= t < FINAL:
        return 'G'
    if BRK <= t < DROP2:
        return BRKP[min(int((t - BRK) // BAR), len(BRKP) - 1)]
    start = DROP2 if t >= DROP2 else 0.0
    return MAIN[int((t - start) // BAR) % 4]

def pent(i, base):
    return base + PEN[i % 5] + 12 * (i // 5)

# ---------------------------------------------------------------- utilidades
B = {k: np.zeros((2, N)) for k in ('kick', 'perc', 'bass', 'pad', 'arp', 'keys', 'sfx', 'send')}

def tt(d):
    return np.arange(int(d * SR)) / SR

def pan2(sig, pan=0.0):
    a = (pan + 1) * np.pi / 4
    return np.vstack([sig * np.cos(a), sig * np.sin(a)]) * np.sqrt(2)

def pan_sweep(sig, p0, p1):
    p = np.linspace(p0, p1, len(sig))
    a = (p + 1) * np.pi / 4
    return np.vstack([sig * np.cos(a), sig * np.sin(a)]) * np.sqrt(2)

def put(bus, t, sig, g=1.0, pan=0.0, send=0.0):
    s = pan2(sig, pan) if sig.ndim == 1 else sig
    i = int(round(t * SR))
    if i < 0:
        s, i = s[:, -i:], 0
    if i >= N or s.shape[1] == 0:
        return
    j = min(N, i + s.shape[1])
    seg = s[:, :j - i] * g
    k = min(int(.004 * SR), seg.shape[1])                    # micro-fade: evita estalos no fim das notas
    seg[:, seg.shape[1] - k:] *= np.linspace(1, 0, k)
    B[bus][:, i:j] += seg
    if send:
        B['send'][:, i:j] += seg * send

def saw(f, n):
    f = np.broadcast_to(np.asarray(f, float), (n,))
    dt = f / SR
    ph = (rng.random() + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m = ph < dt; x = ph[m] / dt[m]; y[m] -= x + x - x * x - 1           # polyBLEP
    m = ph > 1 - dt; x = (ph[m] - 1) / dt[m]; y[m] -= x * x + x + x + 1
    return y

def osc(f, n, ph0=0.0):
    f = np.broadcast_to(np.asarray(f, float), (n,))
    return np.sin(ph0 + 2 * np.pi * np.cumsum(f) / SR)

def _sos(kind, fc, order=2):
    if kind == 'band':
        return signal.butter(order, [fc[0] / (SR / 2), fc[1] / (SR / 2)], 'band', output='sos')
    return signal.butter(order, min(fc, SR * .45) / (SR / 2), kind, output='sos')

def lp(x, fc, order=2): return signal.sosfilt(_sos('low', fc, order), x, axis=-1)
def hp(x, fc, order=2): return signal.sosfilt(_sos('high', fc, order), x, axis=-1)
def bp(x, lo, hi, order=2): return signal.sosfilt(_sos('band', (lo, hi), order), x, axis=-1)

def lp_sweep(x, fcs, C=512):
    """Passa-baixa com corte variável no tempo (mono ou estéreo)."""
    mono = x.ndim == 1
    x = np.atleast_2d(x)
    fcs = np.broadcast_to(np.asarray(fcs, float), (x.shape[1],))
    y = np.empty_like(x)
    zi = None
    for k in range(0, x.shape[1], C):
        sos = _sos('low', float(np.clip(fcs[k], 40, SR * .45)))
        if zi is None:
            zi = np.zeros((x.shape[0], sos.shape[0], 2))
        for c in range(x.shape[0]):
            y[c, k:k + C], zi[c] = signal.sosfilt(sos, x[c, k:k + C], zi=zi[c])
    return y[0] if mono else y

def auto(pts):
    return np.interp(np.arange(N) / SR, [p[0] for p in pts], [p[1] for p in pts])

def fit(parts):
    n = max(p.shape[-1] for p in parts)
    out = np.zeros((2, n))
    for p in parts:
        p = pan2(p) if p.ndim == 1 else p
        out[:, :p.shape[1]] += p
    return out

# ---------------------------------------------------------------- instrumentos
def kick(dec=7.5):
    t = tt(.42)
    f = 46 + 150 * np.exp(-t * 40) + 25 * np.exp(-t * 8)
    body = osc(f, len(t)) * np.exp(-t * dec) * np.minimum(1, t * 900)
    click = np.sin(2 * np.pi * 3000 * t) * np.exp(-t * 1100) * .3
    return np.tanh((body + click) * 1.8) / np.tanh(1.8)

def clap():
    t = tt(.3)
    n = rng.standard_normal(len(t))
    env = sum((t >= d) * np.exp(-np.clip(t - d, 0, None) * 160) for d in (0, .008, .016)) * .6
    env += (t >= .022) * np.exp(-np.clip(t - .022, 0, None) * 26)
    return bp(n, 1000, 3500) * env * .7 + np.sin(2 * np.pi * 185 * t) * np.exp(-t * 45) * .5

def hat(open_=False):
    t = tt(.3 if open_ else .07)
    fr = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * 1.6        # hi-hat metálico (808), sem ruído
    x = sum(np.sign(np.sin(2 * np.pi * f * t + rng.random() * 6.28)) for f in fr)
    y = hp(x, 7000, 4) * np.exp(-t * (10 if open_ else 65)) * np.minimum(1, t * 4000)
    return y / np.abs(y).max()

def tom(f0):
    t = tt(.3)
    return osc(f0 * (1 + .6 * np.exp(-t * 30)), len(t)) * np.exp(-t * 10) * np.minimum(1, t * 800)

_bass = {}
def bass_note(m, d=.22, bright=1.0):
    key = (m, d, bright)
    if key not in _bass:
        t = tt(d); f = mtof(m)
        s = saw(f, len(t)) * .55 + osc(f, len(t)) * .35
        s = lp_sweep(s, 180 + 900 * bright * np.exp(-t * 12))
        sub = osc(f / 2, len(t)) * .7
        env = np.minimum(1, t * 300) * np.exp(-t * 2.2) * np.clip((d - t) / .015, 0, 1)
        _bass[key] = (s + sub) * env
    return _bass[key]

def pad_chord(ms, d):
    t = tt(d); n = len(t)
    L = np.zeros(n); R = np.zeros(n)
    for m in ms:
        for k, c in enumerate((-11, -4, 4, 11)):
            v = saw(mtof(m) * 2 ** (c / 1200), n)
            if k % 2 == 0: L += v * .75; R += v * .25
            else: R += v * .75; L += v * .25
    env = np.minimum(1, t / .18) * np.clip((d - t) / .25, 0, 1)
    return np.vstack([L, R]) * env / (len(ms) * 2)

_pl = {}
def pluck(m):
    if m not in _pl:
        t = tt(.3); f = mtof(m)
        s = saw(f, len(t)) * .6 + osc(2 * f, len(t)) * .2
        s = lp_sweep(s, 700 + 5200 * np.exp(-t * 22))
        _pl[m] = s * np.exp(-t * 13) * np.minimum(1, t * 1500)
    return _pl[m]

def ep(m, d=1.6):
    t = tt(d); f = mtof(m); y = np.zeros(len(t))
    for h, (a, dc) in enumerate([(1, 2.2), (.42, 3.5), (.18, 5), (.1, 7), (.05, 9)], 1):
        y += a * np.sin(2 * np.pi * f * h * t + (0 if h == 1 else rng.random())) * np.exp(-t * dc)
    y += .12 * np.sin(2 * np.pi * f * 7.1 * t) * np.exp(-t * 28)
    return y * np.minimum(1, t * 600)

def stab(ms, d=.7):
    t = tt(d); n = len(t)
    L = np.zeros(n); R = np.zeros(n)
    for m in ms:
        for k, c in enumerate((-9, 0, 9)):
            v = saw(mtof(m) * 2 ** (c / 1200), n)
            L += v * (.8 if k != 2 else .3); R += v * (.8 if k != 0 else .3)
    st = lp_sweep(np.vstack([L, R]) / (len(ms) * 2), 600 + 7000 * np.exp(-t * 9))
    return st * np.exp(-t * 5.5) * np.minimum(1, t * 500)

def bell(m, d=1.4, idx=2.0, ratio=3.5, dec=3.0):
    t = tt(d); f = mtof(m)
    I = idx * np.exp(-t * 7)
    return np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t)) * np.exp(-t * dec) * np.minimum(1, t * 3000)

KICK, KICK_SOFT, CLAP, CHAT, OHAT = kick(), kick(11), clap(), hat(), hat(True)

# ---------------------------------------------------------------- composição
kick_times = []
def kick_at(t, g=1.0, soft=False):
    put('kick', t, KICK_SOFT if soft else KICK, g)
    kick_times.append(t)

def pad_bars(t0, t1):
    t = t0
    while t < t1 - 1e-6:
        d = min(BAR, t1 - t)
        put('pad', t, pad_chord(CH[chord_at(t)], d + .25))
        t += BAR

def arp(t0, t1, g=1.0):
    pat = [0, 1, 2, 3, 2, 1, 2, 3]
    k = 0; t = t0
    while t < t1 - 1e-6:
        ms = [m + 12 for m in CH[chord_at(t)]]
        acc = 1.0 if k % 4 == 0 else .7
        put('arp', t, pluck(ms[pat[k % 8]]), g * acc, pan=.35 * np.sin(k * 1.3))
        k += 1; t += BEAT / 4

def ep_arp(t0, t1, g=1.0):
    pat = [0, 2, 1, 3, 2, 1, 3, 2]
    k = 0; t = t0
    while t < t1 - 1e-6:
        ms = CH[chord_at(t)]
        put('keys', t, ep(ms[pat[k % 8]] + 12), g * (1 if k % 2 == 0 else .75), pan=-.2 + .4 * (k % 2), send=.35)
        k += 1; t += BEAT / 2

def ep_chord(t, g=1.0):
    for m in CH[chord_at(t)]:
        put('keys', t, ep(m, 3.0), g * .5, send=.4)

def groove(t0, t1, lift=False):
    t = t0
    while t < t1 - 1e-6:
        b = int(round(t / BEAT))
        kick_at(t)
        if b % 2 == 1:
            put('perc', t, CLAP, .55, 0, .22)
        for s in range(4):
            ts = t + s * BEAT / 4
            if s == 2:
                put('perc', ts, OHAT, .42, .25)
            else:
                put('perc', ts, CHAT, [.42, .17, 0, .25][s], -.3)
        r = ROOT[chord_at(t)]
        put('bass', t + BEAT / 2, bass_note(r + (12 if b % 4 == 3 else 0)), 1.0)
        put('bass', t + BEAT * .75, bass_note(r, .14, .7), .6)
        if lift and b % 8 == 7:                               # stab no contratempo do 4º tempo
            put('arp', t + BEAT / 2, stab([m + 12 for m in CH[chord_at(t)]], .35), .9, send=.2)
        bar_end = (t + BEAT) % 8 < 1e-6 and (t + BEAT) < t1 - 1e-6
        if bar_end:                                           # virada de tons a cada 4 compassos
            for k, f in enumerate((196, 165, 147, 123)):
                put('perc', t + k * BEAT / 4, tom(f), .45, -.4 + .27 * k, .15)
        t += BEAT

def tom_roll(t0, t1, f0=110, f1=260, g0=.15, g1=.55):
    n = int(round((t1 - t0) / (BEAT / 4)))
    for k in range(n):
        x = k / max(1, n - 1)
        put('perc', t0 + k * BEAT / 4, tom(f0 * (f1 / f0) ** x), g0 + (g1 - g0) * x, .3 * np.sin(k), .15)

# introdução: piano elétrico + pad; o logo (4.6 s) liga o arpejo; build até o drop
ep_arp(0, 4.5, .9)
ep_arp(4.5, DROP1 - .5, .55)
pad_bars(0, DROP1)
arp(4.75, DROP1 - .25, .8)
for t in np.arange(5.0, 7.0, BEAT): kick_at(t, .55, soft=True)
for t in np.arange(7.0, 7.75, BEAT / 2): kick_at(t, .6, soft=True)
tom_roll(6.5, 7.75)

# groove A
groove(DROP1, BRK)
pad_bars(DROP1, BRK)
arp(DROP1, BRK)

# quebra cinematográfica durante o vídeo da matriz
pad_bars(BRK, DROP2)
ep_arp(BRK, DROP2 - 2, 1.0)
for t in np.arange(BRK, DROP2 - 2, BAR):
    put('bass', t, osc(mtof(ROOT[chord_at(t)] - 12), int(BAR * SR)) * np.minimum(1, tt(BAR) * 40) * np.clip((BAR - tt(BAR)) / .1, 0, 1), .8)
for t in (BRK + 4, BRK + 6): kick_at(t, .45, soft=True)
for t in np.arange(DROP2 - 4, DROP2 - .5, BEAT): kick_at(t, .7)
arp(DROP2 - 4, DROP2 - .25, .8)
tom_roll(DROP2 - 1.5, DROP2 - .25)

# groove B até o acorde final
groove(DROP2, FINAL, lift=True)
pad_bars(DROP2, FINAL)
arp(DROP2, FINAL)

# final: acorde de Lá menor que ecoa
put('pad', FINAL, pad_chord(CH['Am'] + [45 + 12], END - FINAL + .3))
ep_chord(FINAL, 1.0)
put('keys', FINAL, bell(93, 3.0, 1.2, 2.0, 1.2), .25, send=.5)

# ---------------------------------------------------------------- efeitos das animações
def sfx_tick(k):
    t = tt(.035); f = 2600 * (1 + .04 * (k % 3))
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 380) + .45 * np.sin(2 * np.pi * 1100 * t) * np.exp(-t * 220)

def sfx_lock(p):
    base = [81, 84, 86, 88, 91, 93][p % 6]
    t = tt(1.3)
    return (bell(base, 1.3) * .6 + bell(base + 7, 1.3, idx=1.5) * .35 + bell(base + 12, 1.3, idx=1.0, dec=5) * .15
            + np.sin(2 * np.pi * 220 * t) * np.exp(-t * 40) * .3)

def sfx_rev(i):
    f = mtof(pent(i, 69)); t = tt(.9)
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t * 7) + .35 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 28)
         + .2 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 12)) * np.minimum(1, t * 1500)
    n = int(.09 * SR); x = np.arange(n) / n
    y[:n] += osc(300 * 3.0 ** x, n) * np.sin(np.pi * x) ** 2 * .25
    return y

def sfx_pop(p):
    t = tt(.12); n = len(t)
    f = 420 * 2 ** (PEN[p % 5] / 12) * (1 + 1.8 * (1 - np.exp(-t * 60)))
    return (osc(f, n) * np.exp(-t * 38) + .3 * osc(2 * f, n) * np.exp(-t * 60)) * np.minimum(1, t * 2000)

def sfx_blip(i):
    f = mtof(min(pent(i, 81), 105)); t = tt(.16)
    return (np.sin(2 * np.pi * f * t) + .22 * np.sin(2 * np.pi * 3 * f * t)) * np.exp(-t * 30) * np.minimum(1, t * 3000)

def sfx_thud(p):
    t = tt(.4)
    f = (95 + 8 * p) * (1 + .9 * np.exp(-t * 28))
    return osc(f, len(t)) * np.exp(-t * 11) * np.minimum(1, t * 900) + np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 260) * .25

def sfx_slide():
    n = int(.28 * SR); t = np.arange(n) / SR; x = np.clip(t / .16, 0, 1); on = (t < .16)
    body = osc(160 * np.exp(-t * 9) + 50, n) * np.exp(-t * 12) * np.minimum(1, t * 600) * .8
    glide = osc(1100 * .35 ** x, n) * np.sin(np.pi * x) ** 2 * on * .22
    air = bp(rng.standard_normal(n), 1500, 5000) * np.sin(np.pi * x) ** 2 * on * .05
    return pan_sweep(body + glide + air, .3, -.3)

def sfx_whip():
    n = int(.16 * SR); x = np.arange(n) / n
    f = 1900 * .14 ** x
    y = (osc(f, n) + .5 * osc(f * 1.5, n)) * np.sin(np.pi * x) ** 1.5 * .5
    y += bp(rng.standard_normal(n), 2000, 7000) * np.sin(np.pi * x) ** 2 * .07
    return pan_sweep(y, -.5, .5)

def sfx_wipe():
    d = .75; n = int(d * SR); x = np.arange(n) / n
    f = 280 + 700 * np.sin(np.pi * x)
    env = np.sin(np.pi * x) ** 2
    y = (osc(f, n) + osc(f * 1.006, n) + .25 * osc(2 * f, n)) * env * .4
    y += bp(rng.standard_normal(n), 1500, 6000) * env * .08
    return pan_sweep(y, -.8, .8)

def sfx_hit(t_at, big):
    ms = CH[chord_at(t_at + .01)]
    st = stab([m + 12 for m in ms] + [ms[0]], .9 if big else .5)
    t = tt(.6)
    sub = osc(40 + 110 * np.exp(-t * 18), len(t)) * np.exp(-t * (6 if big else 10)) * np.minimum(1, t * 1000)
    click = np.sin(2 * np.pi * 3500 * t) * np.exp(-t * 1200) * .3
    return fit([st * (1.0 if big else .6), sub * (.9 if big else .5) + click])

def sfx_draw(d, lo):
    n = int(d * SR); t = np.arange(n) / SR; x = t / d
    f0, f1 = (180, 520) if lo else (420, 1250)
    f = f0 * (f1 / f0) ** x * (1 + .004 * np.sin(2 * np.pi * 6 * t))
    return pan_sweep((osc(f, n) + .3 * osc(2 * f, n)) * np.sin(np.pi * x) ** 1.5, -.6, .6)

def sfx_riser(d):
    n = int(d * SR); x = np.arange(n) / n
    L = np.zeros(n); R = np.zeros(n)
    for m, c in ((57, -8), (64, 8), (69, -5), (76, 5)):
        v = saw(mtof(m) * 2 ** (x * 7 / 12) * 2 ** (c / 1200), n)
        if c < 0: L += v; R += .4 * v
        else: R += v; L += .4 * v
    st = lp_sweep(np.vstack([L, R]) / 4, 300 * 25 ** x) * x ** 2.2
    y = fit([st, osc(55 * 2 ** x, n) * x ** 2 * .5])
    k = int(.012 * SR); y[:, -k:] *= np.linspace(1, 0, k)
    return y

def sfx_shimmer():
    out = np.zeros((2, int(1.8 * SR)))
    for k, m in enumerate([93, 96, 100, 105, 100, 98, 105, 108]):
        s = pan2(bell(m, 1.2, idx=1.2, ratio=2.0, dec=4.5) * .9 ** k, np.sin(k * 1.7) * .7)
        i = int(k * .05 * SR)
        out[:, i:i + s.shape[1]] += s[:, :out.shape[1] - i]
    return out

GAIN = {'tick': .1, 'lock': .3, 'rev': .38, 'pop': .38, 'blip': .25, 'slide': .5, 'whip': .34, 'wipe': .5,
        'hit': .7, 'thud': .55, 'draw': .2, 'riser': .4, 'shimmer': .16}
SEND = {'lock': .35, 'rev': .3, 'pop': .15, 'blip': .3, 'shimmer': .6, 'hit': .25, 'thud': .15, 'draw': .3,
        'riser': .2, 'wipe': .15, 'whip': .1}

rev_i, rev_last = 0, -9
all_cues = CUES + [{'type': 'riser', 't': DROP2 - 2, 'd': 1.95}, {'type': 'hit', 't': FINAL, 'big': 1}]
for c in sorted(all_cues, key=lambda c: c['t']):
    ty, t = c['type'], c['t']
    g = GAIN[ty] * c.get('g', 1)
    if ty == 'tick': s = sfx_tick(c.get('k', 0)); pan = .25 * (1 if c.get('k', 0) % 2 else -1)
    elif ty == 'lock': s = sfx_lock(c.get('p', 0)); pan = 0
    elif ty == 'rev':
        if 'i' in c: i = c['i']
        else:
            i = rev_i + 1 if t - rev_last < 1.0 else 0
        rev_i, rev_last = i, t
        s = sfx_rev(i); pan = -.2 + .1 * (i % 5)
    elif ty == 'pop': s = sfx_pop(c.get('p', 0)); pan = -.3 + .15 * (c.get('p', 0) % 5)
    elif ty == 'blip': s = sfx_blip(c.get('i', 0)); pan = .5 * np.sin(c.get('i', 0) * 1.1)
    elif ty == 'thud': s = sfx_thud(c.get('p', 0)); pan = -.4 + .4 * (c.get('p', 0) % 3)
    elif ty == 'slide': s = sfx_slide(); pan = 0
    elif ty == 'whip': s = sfx_whip(); pan = 0
    elif ty == 'wipe': s = sfx_wipe(); pan = 0
    elif ty == 'hit': s = sfx_hit(t, bool(c.get('big'))); g *= 1 if c.get('big') else .65; pan = 0
    elif ty == 'draw': s = sfx_draw(c.get('d', 1), c.get('lo', 0)); pan = 0
    elif ty == 'riser': s = sfx_riser(c.get('d', 1.5)); pan = 0
    elif ty == 'shimmer': s = sfx_shimmer(); pan = 0
    else: continue
    put('sfx', t, s, g, pan, SEND.get(ty, 0))

# ---------------------------------------------------------------- mixagem
def active_rms(x):
    e = np.sqrt(np.mean(x ** 2, axis=0))
    w = int(.4 * SR); m = len(e) // w
    blocks = np.sqrt(np.mean(e[:m * w].reshape(m, w) ** 2, axis=1))
    on = blocks[blocks > blocks.max() * .05]
    return float(np.sqrt(np.mean(on ** 2))) if len(on) else 1.0

def db(x): return 10 ** (x / 20)

# sidechain (a música "respira" com o bumbo)
imp = np.zeros(N)
for tk in kick_times:
    i = int(tk * SR)
    if i < N: imp[i] = 1
env = signal.lfilter([1], [1, -np.exp(-1 / (.13 * SR))], imp)
env = lp(np.clip(env, 0, 1), 90)
duck = lambda depth: 1 - depth * env

B['pad'] = lp_sweep(B['pad'], auto([(0, 500), (4.5, 1100), (4.6, 1800), (DROP1 - .1, 4200), (DROP1, 4000), (BRK - .01, 4000),
                                    (BRK, 1400), (DROP2 - 4, 1600), (DROP2 - .05, 5500), (DROP2, 4200), (FINAL, 4200), (END, 1800)]))
B['pad'] *= auto([(0, 0), (1.2, .8), (4.6, .9), (DROP1, 1), (BRK, 1), (BRK + .01, 1.3), (DROP2, 1.3), (DROP2 + .01, 1),
                  (FINAL, 1), (FINAL + .01, 1.3), (END, 1.3)])
B['arp'] = lp_sweep(B['arp'], auto([(0, 1500), (4.6, 1500), (DROP1 - .1, 7000), (DROP1, 6500), (DROP2 - 4, 900),
                                    (DROP2 - .05, 7000), (DROP2, 6500), (END, 6500)]))
m = (B['arp'][0] + B['arp'][1]) / 2                          # delay ping-pong em colcheia pontuada
d = int(.375 * SR); mf = lp(m, 3500)
for k, gk in enumerate((.36, .22, .12), 1):
    ch = k % 2
    B['arp'][ch, k * d:] += gk * mf[:-k * d]
B['send'] += B['arp'] * .15 + B['pad'] * .1

targets = {'bass': -19, 'pad': -24, 'arp': -26, 'keys': -23}
for k, v in targets.items():
    B[k] *= db(v) / active_rms(B[k])
B['kick'] *= .8 / np.abs(B['kick']).max()
B['perc'] *= .33 / np.abs(B['perc']).max()
B['sfx'] *= .95 / np.abs(B['sfx']).max()

def reverb(x, dur=2.4, pre=.02):
    n = int(dur * SR); t = np.arange(n) / SR
    out = []
    for c in range(2):
        ir = rng.standard_normal(n) * np.exp(-t * (6.9 / dur))
        ir = lp_sweep(ir, 800 + 9000 * np.exp(-t * 1.2))
        ir[:int(pre * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        out.append(signal.fftconvolve(x[c], ir)[:x.shape[1]])
    return np.vstack(out)

wet = reverb(B['send'])
wet *= db(-27) / active_rms(wet)

mix = (B['kick'] + B['perc'] + B['bass'] * duck(.75) + B['pad'] * duck(.5) + B['arp'] * duck(.35)
       + B['keys'] + wet * duck(.3) + B['sfx'])
mix = hp(mix, 28)

# compressão suave + limitador com lookahead
e = np.sqrt(lp(np.mean(mix ** 2, axis=0), 8) + 1e-12)
thr = db(-15)
gr = np.where(e > thr, (e / thr) ** (1 / 2 - 1), 1.0)
mix *= lp(gr, 20)
target = db(-14.5) / active_rms(mix)
mix *= target
ceil = .93
la = int(.003 * SR)
pk = maximum_filter1d(np.max(np.abs(mix), axis=0), size=2 * la + 1)
g = np.minimum(1.0, ceil / (pk + 1e-9))
g = minimum_filter1d(g, size=2 * la + 1)
g = signal.lfilter([1 - np.exp(-1 / (.06 * SR))], [1, -np.exp(-1 / (.06 * SR))], g - 1) + 1
g = np.minimum(g, np.minimum(1.0, ceil / (pk + 1e-9)))
mix *= g
mix = np.tanh(mix / ceil * 1.02) * ceil

L = int(END * SR)
mix = mix[:, :L]
fade = np.ones(L); f = int(1.6 * SR); fade[-f:] = np.linspace(1, 0, f) ** 1.5
fade[:int(.05 * SR)] = np.linspace(0, 1, int(.05 * SR))
mix *= fade

import os
if os.environ.get('STEMS'):
    import wave as _w
    for nm, x in (('sfx', B['sfx']), ('music', B['kick'] + B['perc'] + B['bass'] * duck(.75) + B['pad'] * duck(.5) + B['arp'] * duck(.35) + B['keys'])):
        with _w.open(OUT.replace('.wav', f'_{nm}.wav'), 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(x[:, :L].T * target, -1, 1) * 32767).astype('<i2').tobytes())

import wave
with wave.open(OUT, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix.T, -1, 1) * 32767).astype('<i2').tobytes())
rms = 20 * np.log10(active_rms(mix)); peak = 20 * np.log10(np.abs(mix).max())
print(f'ok {OUT}: {END}s  rms {rms:.1f} dBFS  pico {peak:.1f} dBFS  deixas {len(CUES)}  bumbos {len(kick_times)}')
