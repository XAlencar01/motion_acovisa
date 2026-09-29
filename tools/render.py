"""Renderiza o index.html quadro a quadro (30 fps) e monta o MP4 com a trilha.

Uso: python3 tools/render.py --drone VIDEO_DRONE.mp4 --out acovisa_motion_reel.mp4
  --drone  vídeo original do drone (1080p). Sem ele, usa assets/matriz_guarulhos.mp4 (720p).
"""
import argparse, asyncio, json, os, pathlib, subprocess, sys, tempfile
import imageio_ffmpeg
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
FF = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 30

ap = argparse.ArgumentParser()
ap.add_argument('--drone', default=str(ROOT / 'assets' / 'matriz_guarulhos.mp4'))
ap.add_argument('--out', default=str(ROOT / 'acovisa_motion_reel.mp4'))
ap.add_argument('--work', default=tempfile.mkdtemp(prefix='reel_'))
ap.add_argument('--workers', type=int, default=6)
A = ap.parse_args()
work = pathlib.Path(A.work); (work / 'fr').mkdir(parents=True, exist_ok=True); (work / 'drone').mkdir(exist_ok=True)

subprocess.run([FF, '-loglevel', 'error', '-y', '-i', A.drone, '-vf', f'fps={FPS},scale=1920:1080', '-q:v', '2',
                str(work / 'drone' / '%04d.jpg')], check=True)
fn = len(list((work / 'drone').glob('*.jpg')))
url = (ROOT / 'index.html').as_uri() + f'?render&frames={(work / "drone").as_uri()}/&fn={fn}'

async def worker(a, b):
    async with async_playwright() as p:
        br = await p.chromium.launch(executable_path=CHROME, args=['--allow-file-access-from-files'])
        pg = await br.new_page(viewport={'width': 1920, 'height': 1080})
        await pg.goto(url)
        await pg.evaluate('document.fonts.ready'); await pg.wait_for_timeout(600)
        for i in range(a, b):
            await pg.evaluate(f'seek({i / FPS})')
            await pg.screenshot(path=str(work / 'fr' / f'{i:05d}.jpg'), type='jpeg', quality=92)
        await br.close()

async def main():
    async with async_playwright() as p:
        br = await p.chromium.launch(executable_path=CHROME)
        pg = await br.new_page(); await pg.goto(url)
        data = await pg.evaluate('({cues:window.CUES,marks:window.MARKS})')
        await br.close()
    json.dump(data, open(work / 'cues.json', 'w'))
    total = int(data['marks']['END'] * FPS)
    step = -(-total // A.workers)
    await asyncio.gather(*[worker(k * step, min(total, (k + 1) * step)) for k in range(A.workers)])
    return total

total = asyncio.run(main())
subprocess.run([sys.executable, str(ROOT / 'tools' / 'soundtrack.py'), str(work / 'cues.json'), str(work / 'trilha.wav')], check=True)
subprocess.run([FF, '-loglevel', 'error', '-y', '-framerate', str(FPS), '-i', str(work / 'fr' / '%05d.jpg'), '-i', str(work / 'trilha.wav'),
                '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                '-shortest', '-movflags', '+faststart', A.out], check=True)
subprocess.run([FF, '-loglevel', 'error', '-y', '-i', str(work / 'trilha.wav'), '-c:a', 'aac', '-b:a', '160k',
                str(ROOT / 'assets' / 'soundtrack.m4a')], check=True)
subprocess.run([FF, '-loglevel', 'error', '-y', '-i', str(work / 'trilha.wav'), '-c:a', 'libopus', '-b:a', '128k',
                str(ROOT / 'assets' / 'soundtrack.ogg')], check=True)
print('ok', A.out, total, 'quadros')
