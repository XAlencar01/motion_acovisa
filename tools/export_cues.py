"""Exporta as deixas de som (CUES) e as marcações de cena do index.html para JSON."""
import asyncio, json, sys, pathlib
from playwright.async_api import async_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'tools' / 'cues.json')
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME)
        pg = await b.new_page()
        await pg.goto((ROOT / 'index.html').as_uri() + '?render')
        data = await pg.evaluate('({cues:window.CUES,marks:window.MARKS})')
        await b.close()
    json.dump(data, open(out, 'w'), indent=0)
    print(len(data['cues']), 'cues ->', out)
asyncio.run(main())
