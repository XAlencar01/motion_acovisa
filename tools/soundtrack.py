"""Trilha institucional do reel Açovisa: música original (Dó maior, 120 BPM) + desenho de som das animações.

Música: piano em ostinato, cordas, pulso de baixo e percussão leve (bumbo, palmas, shaker).
Efeitos: sopros de ar filtrados nas transições, impactos graves e curtos com acorde de piano,
toques discretos em cards e ícones, ticks baixos nos contadores e swells nos risers.
Sem notas melódicas "de jogo", sem sininhos e sem estrondos longos.

As deixas vêm do próprio index.html (window.CUES), então cada animação tem o seu som no tempo exato.
Uso: python3 tools/soundtrack.py cues.json saida.wav
"""
import json
import os
import sys
import wave

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
LIFT = TS['s8']                              # a partir daqui as cordas ganham a oitava de cima
FINAL = 2 * round((OUTRO + 5) / 2)          # acorde final, no início de um compasso

# ---------------------------------------------------------------- harmonia
def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)

CH = {'C': [55, 60, 64, 67], 'G': [55, 59, 62, 67], 'Am': [57, 60, 64, 69], 'F': [53, 57, 60, 65],
      'Cadd9': [55, 60, 62, 64, 67]}
ROOT = {'C': 48, 'G': 43, 'Am': 45, 'F': 41, 'Cadd9': 48}
MAIN = ['C', 'G', 'Am', 'F']
BRKP = ['Am', 'F', 'C', 'G', 'F', 'G']      # quebra: termina em G para resolver em C no drop

def chord_at(t):
    if t >= FINAL:
        return 'Cadd9'
    if FINAL - BAR <= t < FINAL:
        return 'G'
    if BRK <= t < DROP2:
        return BRKP[min(int((t - BRK) // BAR), len(BRKP) - 1)]
    start = DROP2 if t >= DROP2 else 0.0
    return MAIN[int((t - start) // BAR) % 4]

# ---------------------------------------------------------------- utilidades
B = {k: np.zeros((2, N)) for k in ('kick', 'perc', 'bass', 'strings', 'piano', 'sfx', 'send')}

def tt(d):
    return np.arange(int(d * SR)) / SR

def pan2(sig, pan=0.0):
    a = (pan + 1) * np.pi / 4
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
    k = min(int(.004 * SR), seg.shape[1])                    # micro-fade: evita estalos
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

def noise(n):
    return rng.standard_normal(n)

def _sos(kind, fc, order=2):
    if kind == 'band':
        return signal.butter(order, [fc[0] / (SR / 2), fc[1] / (SR / 2)], 'band', output='sos')
    return signal.butter(order, min(fc, SR * .45) / (SR / 2), kind, output='sos')

def lp(x, fc, order=2): return signal.sosfilt(_sos('low', fc, order), x, axis=-1)
def hp(x, fc, order=2): return signal.sosfilt(_sos('high', fc, order), x, axis=-1)
def bp(x, lo, hi, order=2): return signal.sosfilt(_sos('band', (lo, hi), order), x, axis=-1)

def sweep(x, fcs, kind='low', q=1.4, C=256):
    """Filtro com frequência variável no tempo (passa-baixa ou passa-banda), mono ou estéreo."""
    mono = x.ndim == 1
    x = np.atleast_2d(x)
    fcs = np.broadcast_to(np.asarray(fcs, float), (x.shape[1],))
    y = np.empty_like(x); zi = None
    for k in range(0, x.shape[1], C):
        fc = float(np.clip(fcs[k], 40, SR * .4))
        if kind == 'band':
            w = 1 / (2 * q)
            sos = _sos('band', (fc * (1 - w), min(fc * (1 + w), SR * .45)))
        else:
            sos = _sos('low', fc)
        if zi is None:
            zi = np.zeros((x.shape[0], sos.shape[0], 2))
        for c in range(x.shape[0]):
            y[c, k:k + C], zi[c] = signal.sosfilt(sos, x[c, k:k + C], zi=zi[c])
    return y[0] if mono else y

def auto(pts):
    return np.interp(np.arange(N) / SR, [p[0] for p in pts], [p[1] for p in pts])

def norm(x):
    return x / (np.abs(x).max() + 1e-9)

def mix2(parts):
    n = max(p.shape[-1] for p in parts)
    out = np.zeros((2, n))
    for p in parts:
        p = pan2(p) if p.ndim == 1 else p
        out[:, :p.shape[1]] += p
    return out

# ---------------------------------------------------------------- instrumentos
def kick(dec=6.5, g_click=.12):
    t = tt(.45)
    f = 44 + 110 * np.exp(-t * 35) + 18 * np.exp(-t * 8)
    body = osc(f, len(t)) * np.exp(-t * dec) * np.minimum(1, t * 700)
    click = lp(noise(len(t)), 4000) * np.exp(-t * 700) * g_click
    return np.tanh((body + click) * 1.5) / np.tanh(1.5)

def clap():
    t = tt(.35)
    env = sum((t >= d) * np.exp(-np.clip(t - d, 0, None) * 140) for d in (0, .01, .02)) * .5
    env += (t >= .026) * np.exp(-np.clip(t - .026, 0, None) * 20)
    return lp(bp(noise(len(t)), 900, 2600), 3000) * env + np.sin(2 * np.pi * 180 * t) * np.exp(-t * 40) * .35

def snare():
    t = tt(.22)
    return lp(bp(noise(len(t)), 1200, 4200), 5000) * np.exp(-t * 26) * .8 + osc(190 * (1 + .2 * np.exp(-t * 40)), len(t)) * np.exp(-t * 30) * .5

def shaker(k):
    t = tt(.07)
    return bp(noise(len(t)), 4500, 9000) * np.exp(-t * 60) * np.minimum(1, t / .006)

_pn = {}
def piano(m, vel=.8, d=2.2):
    """Piano sintetizado: parciais levemente inarmônicas, duas cordas por nota e martelo suave."""
    vel = round(vel, 2)
    key = (m, vel, d)
    if key not in _pn:
        t = tt(d); n = len(t); f = mtof(m); y = np.zeros(n)
        for h in range(1, 12):
            fh = f * h * np.sqrt(1 + .00035 * h * h)
            if fh > SR * .42:
                break
            a = (1 / h ** 1.15) * (.35 + .65 * vel) ** (h * .5)
            dec = .55 + .32 * h + f / 900
            for c in (-.8, .8):
                y += a * .5 * np.sin(2 * np.pi * fh * 2 ** (c / 1200) * t + rng.random() * 6.28) * np.exp(-t * dec)
        k = int(.006 * SR)
        y[:k] += lp(noise(k), 2200) * np.hanning(k) * .12 * vel
        y *= np.minimum(1, t * 900) * np.clip((d - t) / .25, 0, 1)
        _pn[key] = norm(y) * vel
    return _pn[key]

def strings(ms, d, octave_up=False):
    """Naipe de cordas: serras desafinadas com vibrato lento, ataque macio e violoncelo uma oitava abaixo."""
    t = tt(d); n = len(t)
    L = np.zeros(n); R = np.zeros(n)
    notes = list(ms) + ([ms[-1] + 12, ms[-2] + 12] if octave_up else [])
    for m in notes:
        for k, c in enumerate((-12, -5, 5, 12)):
            vib = 1 + .0018 * np.sin(2 * np.pi * (4.8 + .3 * k) * t + rng.random() * 6.28)
            v = saw(mtof(m) * 2 ** (c / 1200) * vib, n)
            if k % 2 == 0: L += v * .8; R += v * .3
            else: R += v * .8; L += v * .3
    cello = saw(mtof(ms[0] - 12) * (1 + .0015 * np.sin(2 * np.pi * 4.6 * t)), n) * 1.2
    L += cello; R += cello
    env = np.minimum(1, t / .45) ** 1.5 * np.clip((d - t) / .4, 0, 1)
    return np.vstack([L, R]) * env / (len(notes) * 2)

_bs = {}
def bass_pulse(m, d=.24):
    if (m, d) not in _bs:
        t = tt(d); f = mtof(m)
        s = sweep(saw(f, len(t)) * .5 + osc(f, len(t)) * .5, 140 + 520 * np.exp(-t * 10))
        env = np.minimum(1, t * 250) * np.exp(-t * 2) * np.clip((d - t) / .02, 0, 1)
        _bs[(m, d)] = (s + osc(f / 2, len(t)) * .6) * env
    return _bs[(m, d)]

KICK, KICK_SOFT, CLAP, SNARE = kick(), kick(10, .05), clap(), snare()
SHAKE = [shaker(k) for k in range(4)]

# ---------------------------------------------------------------- composição
kick_times = []
def kick_at(t, g=1.0, soft=False):
    put('kick', t, KICK_SOFT if soft else KICK, g)
    kick_times.append(t)

def strings_bars(t0, t1, octave_up=False):
    t = t0
    while t < t1 - 1e-6:
        d = min(BAR, t1 - t)
        put('strings', t, strings(CH[chord_at(t)], d + .4, octave_up))
        t += BAR

def piano_ostinato(t0, t1, vel=.75, step=BEAT / 2):
    """Colcheias em acorde quebrado (padrão corporativo), com acento no tempo."""
    pat = [0, 2, 1, 3, 2, 1, 3, 2]
    k = 0; t = t0
    while t < t1 - 1e-6:
        ms = [m + 12 for m in CH[chord_at(t)]][:4]
        v = vel * (1 if k % 4 == 0 else .78)
        put('piano', t, piano(ms[pat[k % 8] % len(ms)], v), 1.0, pan=-.25 + .5 * ((k % 4) / 3), send=.25)
        if k % 8 == 0:                                        # nota grave no início do compasso
            put('piano', t, piano(ROOT[chord_at(t)], vel * .9, 2.6), .8, send=.25)
        k += 1; t += step

def piano_chord(t, vel=.9, low=True):
    ch = chord_at(t + .01)
    ms = CH[ch] + ([ROOT[ch], ROOT[ch] - 12] if low else [])
    for m in ms:
        put('piano', t, piano(m, vel, 3.2), .55, send=.35)

def groove(t0, t1):
    t = t0
    while t < t1 - 1e-6:
        b = int(round(t / BEAT))
        kick_at(t)
        if b % 2 == 1:
            put('perc', t, CLAP, .5, 0, .3)
        for s in range(4):
            put('perc', t + s * BEAT / 4, SHAKE[s], [.42, .2, .32, .2][s], .35)
        r = ROOT[chord_at(t)]
        put('bass', t, bass_pulse(r), .8)
        put('bass', t + BEAT / 2, bass_pulse(r), 1.0)
        if (t + BEAT) % 16 < 1e-6 and t + BEAT < t1 - 1e-6:  # respiro a cada 8 compassos
            for k in range(2):
                put('perc', t + BEAT / 2 + k * BEAT / 4, SNARE, .35 + .15 * k, 0, .25)
        t += BEAT

def snare_roll(t0, t1, g0=.08, g1=.5):
    n16 = int(round((t1 - t0) / (BEAT / 4)))
    half = n16 // 2
    times = [t0 + k * BEAT / 4 for k in range(half)]
    t = t0 + half * BEAT / 4
    while t < t1 - 1e-6:
        times.append(t); t += BEAT / 8
    for k, ts in enumerate(times):
        x = k / max(1, len(times) - 1)
        put('perc', ts, SNARE, g0 + (g1 - g0) * x ** 1.6, 0, .25)

# introdução: piano solo e cordas; o logo (4.6 s) abre a textura; build até o drop
piano_ostinato(0, DROP1 - .25, .55)
strings_bars(0, DROP1)
for t in np.arange(5.0, 7.5, BEAT): kick_at(t, .5, soft=True)
snare_roll(6.5, 7.75)

# groove A
groove(DROP1, BRK)
strings_bars(DROP1, BRK)
piano_ostinato(DROP1, BRK, .8)

# quebra durante o vídeo da matriz: cordas, piano em semínimas, baixo sustentado
strings_bars(BRK, DROP2)
piano_ostinato(BRK, DROP2 - 4, .6, step=BEAT)
for t in np.arange(BRK, DROP2 - 2, BAR):
    s = osc(mtof(ROOT[chord_at(t)] - 12), int(BAR * SR)) * np.minimum(1, tt(BAR) * 30) * np.clip((BAR - tt(BAR)) / .1, 0, 1)
    put('bass', t, s, .7)
for t in (BRK + 4, BRK + 6): kick_at(t, .45, soft=True)
piano_ostinato(DROP2 - 4, DROP2 - .25, .7)
for t in np.arange(DROP2 - 4, DROP2 - .5, BEAT): kick_at(t, .65, soft=True)
snare_roll(DROP2 - 1.5, DROP2 - .25)

# groove B (cordas com oitava de cima a partir da montagem dos processos)
groove(DROP2, FINAL)
strings_bars(DROP2, LIFT)
strings_bars(LIFT, FINAL, octave_up=True)
piano_ostinato(DROP2, FINAL, .82)

# final: Dó com nona, cordas e piano que ecoam
put('strings', FINAL, strings(CH['Cadd9'], END - FINAL + .4, True))
piano_chord(FINAL, .9)

# ---------------------------------------------------------------- desenho de som das animações
def pan_env(y, p0, p1):
    y = np.atleast_2d(y)
    if y.shape[0] == 1:
        y = np.vstack([y[0], y[0]])
    a = (np.linspace(p0, p1, y.shape[1]) + 1) * np.pi / 4
    return np.vstack([y[0] * np.cos(a), y[1] * np.sin(a)]) * np.sqrt(2)

def swoosh(d, f0, f1, f2, body=.4, peak=.55, p0=-.6, p1=.6):
    """Sopro de ar: ruído em passa-banda que varre a frequência, com um pouco de corpo grave."""
    n = int(d * SR); x = np.arange(n) / n
    env = np.where(x < peak, (x / peak) ** 2.2, ((1 - x) / (1 - peak)) ** 1.7)
    cs = np.where(x < peak, f0 * (f1 / f0) ** (x / peak), f1 * (f2 / f1) ** ((x - peak) / (1 - peak)))
    st = np.vstack([sweep(noise(n), cs, 'band', 1.3), sweep(noise(n), cs * 1.06, 'band', 1.3)])
    st = lp(st, 6500) * env
    st = st / (np.abs(st).max() + 1e-9)
    lo = lp(noise(n), 200) * env
    st += pan2(lo / (np.abs(lo).max() + 1e-9) * body * .6, 0)
    return pan_env(norm(st), p0, p1)

def sfx_tick():
    t = tt(.02)
    return bp(noise(len(t)), 1400, 3800) * np.exp(-t * 450) + np.sin(2 * np.pi * 1700 * t) * np.exp(-t * 520) * .25

def sfx_tap():
    t = tt(.14); n = len(t)
    return (bp(noise(n), 700, 2400) * np.exp(-t * 170) * .5
            + osc(200 * (1 + .12 * np.exp(-t * 50)), n) * np.exp(-t * 40) * .8)

def sfx_settle():
    t = tt(.4); n = len(t)
    return (osc(72 * (1 + .25 * np.exp(-t * 30)), n) * np.exp(-t * 13) * np.minimum(1, t * 600)
            + lp(noise(n), 700) * np.exp(-t * 40) * .25)

def sfx_thud():
    t = tt(.5); n = len(t)
    return (osc(82 * (1 + .18 * np.exp(-t * 30)), n) * np.exp(-t * 10) * np.minimum(1, t * 700)
            + lp(noise(n), 650) * np.exp(-t * 32) * .45 + bp(noise(n), 900, 3000) * np.exp(-t * 200) * .15)

def sfx_impact(big):
    t = tt(1.1 if big else .6); n = len(t)
    sub = osc(38 + 64 * np.exp(-t * 14), n) * np.exp(-t * (4.8 if big else 8)) * np.minimum(1, t * 600)
    body = lp(noise(n), 900) * np.exp(-t * (10 if big else 15)) * np.minimum(1, t * 1500) * .45
    tom = osc(88 * (1 + .2 * np.exp(-t * 20)), n) * np.exp(-t * 7) * .45
    trans = bp(noise(n), 1500, 5000) * np.exp(-t * 130) * .12
    return norm(sub * .9 + body + tom + trans)

def sfx_swell(d, tonal_at=None):
    """Swell sem subida de tom: ruído com o filtro abrindo + acorde de cordas em crescendo."""
    n = int(d * SR); x = np.arange(n) / n
    air = np.vstack([sweep(noise(n), 250 * 24 ** x), sweep(noise(n), 260 * 24 ** x)]) * x ** 2.4
    air = norm(air) * .55
    parts = [air]
    if tonal_at is not None:
        st = sweep(strings(CH[chord_at(tonal_at)], d), 400 * 12 ** x) * x ** 2
        parts.append(norm(st) * .7)
    y = mix2(parts)
    k = int(.015 * SR); y[:, -k:] *= np.linspace(1, 0, k)
    return y

GAIN = {'tick': .07, 'blip': .045, 'lock': .3, 'rev': .15, 'pop': .26, 'slide': .3, 'whip': .24, 'wipe': .36,
        'hit': .65, 'thud': .4, 'draw': .1, 'riser': .36}
SEND = {'hit': .3, 'thud': .2, 'pop': .08, 'slide': .12, 'whip': .12, 'wipe': .12, 'lock': .15, 'riser': .25, 'rev': .1, 'draw': .2}
GAP = {'tick': .045, 'blip': .07, 'rev': .25, 'pop': .08}   # espaçamento mínimo: evita metralhadora de sons

last = {}
all_cues = CUES + [{'type': 'riser', 't': DROP2 - 2, 'd': 1.95}, {'type': 'hit', 't': FINAL, 'big': 1}]
for c in sorted(all_cues, key=lambda c: c['t']):
    ty, t = c['type'], c['t']
    if ty not in GAIN:
        continue                                            # 'shimmer' e afins: fora do som institucional
    if t - last.get(ty, -9) < GAP.get(ty, 0):
        continue
    last[ty] = t
    g = GAIN[ty] * c.get('g', 1); pan = 0.0
    if ty in ('tick', 'blip'):
        s = sfx_tick(); pan = .2 * np.sin(t * 7)
    elif ty == 'lock':
        s = sfx_settle()
    elif ty == 'rev':
        s = swoosh(.3, 1100, 3400, 2200, body=0, peak=.4, p0=-.25, p1=.25)
    elif ty == 'pop':
        s = sfx_tap(); pan = -.2 + .1 * (c.get('p', 0) % 5)
    elif ty == 'slide':
        s = swoosh(.45, 300, 1700, 600, body=.5, p0=.4, p1=-.4)
    elif ty == 'whip':
        s = swoosh(.32, 600, 2800, 1200, body=.2, p0=-.5, p1=.5)
    elif ty == 'wipe':
        s = swoosh(.8, 250, 2100, 700, body=.6, p0=-.8, p1=.8)
    elif ty == 'hit':
        big = bool(c.get('big'))
        s = sfx_impact(big); g *= 1 if big else .6
        piano_chord(t, .95 if big else .7, low=True)       # o impacto ganha um acorde de piano
    elif ty == 'thud':
        s = sfx_thud(); pan = -.3 + .3 * (c.get('p', 0) % 3)
    elif ty == 'draw':
        s = sfx_swell(c.get('d', 1))
    elif ty == 'riser':
        s = sfx_swell(c.get('d', 1.5), tonal_at=t + c.get('d', 1.5) + .05)
    put('sfx', t, s, g, pan, SEND.get(ty, 0))

# ---------------------------------------------------------------- mixagem
def active_rms(x):
    e = np.sqrt(np.mean(x ** 2, axis=0))
    w = int(.4 * SR); m = len(e) // w
    blocks = np.sqrt(np.mean(e[:m * w].reshape(m, w) ** 2, axis=1))
    on = blocks[blocks > blocks.max() * .05]
    return float(np.sqrt(np.mean(on ** 2))) if len(on) else 1.0

def db(x): return 10 ** (x / 20)

imp = np.zeros(N)                                           # sidechain: a música respira com o bumbo
for tk in kick_times:
    i = int(tk * SR)
    if i < N: imp[i] = 1
env = signal.lfilter([1], [1, -np.exp(-1 / (.13 * SR))], imp)
env = lp(np.clip(env, 0, 1), 90)
duck = lambda depth: 1 - depth * env

B['strings'] = sweep(B['strings'], auto([(0, 900), (4.5, 1500), (4.6, 2400), (DROP1 - .1, 4200), (DROP1, 3200),
                                         (BRK, 2600), (DROP2 - .05, 4500), (DROP2, 3400), (END, 3000)]), C=512)
B['strings'] *= auto([(0, 0), (1.5, .7), (4.6, .85), (DROP1, 1), (BRK, 1), (BRK + .01, 1.25), (DROP2, 1.25),
                      (DROP2 + .01, 1), (FINAL, 1), (FINAL + .01, 1.2), (END, 1.2)])
B['piano'] = sweep(B['piano'], auto([(0, 3500), (4.6, 4500), (DROP1, 7000), (END, 7000)]), C=512)
B['send'] += B['strings'] * .12 + B['piano'] * .05

targets = {'bass': -20, 'strings': -22, 'piano': -22}
for k, v in targets.items():
    B[k] *= db(v) / active_rms(B[k])
B['kick'] *= .72 / np.abs(B['kick']).max()
B['perc'] *= .3 / np.abs(B['perc']).max()
B['sfx'] *= .9 / np.abs(B['sfx']).max()

def reverb(x, dur=2.8, pre=.025):
    n = int(dur * SR); t = np.arange(n) / SR
    out = []
    for c in range(2):
        ir = noise(n) * np.exp(-t * (6.9 / dur))
        ir = sweep(ir, 700 + 8000 * np.exp(-t * 1.4), C=512)
        ir[:int(pre * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        out.append(signal.fftconvolve(x[c], ir)[:x.shape[1]])
    return np.vstack(out)

wet = reverb(B['send'])
wet *= db(-25) / active_rms(wet)

music = B['kick'] + B['perc'] + B['bass'] * duck(.7) + B['strings'] * duck(.4) + B['piano'] * duck(.2)
mix = hp(music + wet * duck(.25) + B['sfx'], 28)

# compressão suave + limitador com lookahead
e = np.sqrt(lp(np.mean(mix ** 2, axis=0), 8) + 1e-12)
thr = db(-15)
mix *= lp(np.where(e > thr, (e / thr) ** -.5, 1.0), 20)
target = db(-14.5) / active_rms(mix)
mix *= target
ceil = .93
la = int(.003 * SR)
pk = maximum_filter1d(np.max(np.abs(mix), axis=0), size=2 * la + 1)
g = minimum_filter1d(np.minimum(1.0, ceil / (pk + 1e-9)), size=2 * la + 1)
g = signal.lfilter([1 - np.exp(-1 / (.06 * SR))], [1, -np.exp(-1 / (.06 * SR))], g - 1) + 1
mix *= np.minimum(g, np.minimum(1.0, ceil / (pk + 1e-9)))
mix = np.tanh(mix / ceil * 1.02) * ceil

L = int(END * SR)
mix = mix[:, :L]
fade = np.ones(L); f = int(1.8 * SR); fade[-f:] = np.linspace(1, 0, f) ** 1.5
fade[:int(.05 * SR)] = np.linspace(0, 1, int(.05 * SR))
mix *= fade

def write(path, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x.T, -1, 1) * 32767).astype('<i2').tobytes())

write(OUT, mix)
if os.environ.get('STEMS'):
    write(OUT.replace('.wav', '_sfx.wav'), B['sfx'][:, :L] * target)
    write(OUT.replace('.wav', '_music.wav'), music[:, :L] * target)
rms = 20 * np.log10(active_rms(mix)); peak = 20 * np.log10(np.abs(mix).max())
print(f'ok {OUT}: {END}s  rms {rms:.1f} dBFS  pico {peak:.1f} dBFS  deixas {len(CUES)}  bumbos {len(kick_times)}')
