"""Download the trained models from the GitHub release into checkpoints/ (no account needed).

    python scripts/download_models.py                  # every model (~500 MB)
    python scripts/download_models.py rhoa_web2.pt     # only the ones you name
    python scripts/download_models.py --tag models-v1 --out checkpoints

Files already downloaded are skipped. Each .pt is checked against the sha256 in MODELS.txt.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = 'benisalla/tiny-dlms'


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'tiny-dlms-download'})
    return urllib.request.urlopen(req)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*', help='files to download (default: all)')
    parser.add_argument('--tag', default='models-v1')
    parser.add_argument('--out', default='checkpoints')
    parser.add_argument('--repo', default=REPO)
    args = parser.parse_args()

    release = json.load(get(f'https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}'))
    assets = {a['name']: a['browser_download_url'] for a in release['assets']}
    expected = {}
    if 'MODELS.txt' in assets:
        for line in get(assets['MODELS.txt']).read().decode().splitlines():
            digest, _size, name = line.split(maxsplit=2)
            expected[name] = digest

    names = args.names or sorted(n for n in assets if n != 'MODELS.txt')
    missing = [n for n in names if n not in assets]
    if missing:
        sys.exit(f"not in release {args.tag}: {', '.join(missing)}")

    os.makedirs(args.out, exist_ok=True)
    for name in names:
        path = os.path.join(args.out, name)
        if os.path.exists(path) and (name not in expected or sha256(path) == expected[name]):
            print(f'{name}: already here')
            continue
        print(f'{name}: downloading')
        with get(assets[name]) as r, open(path + '.part', 'wb') as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(path + '.part', path)
        if name in expected and sha256(path) != expected[name]:
            os.remove(path)
            sys.exit(f'{name}: sha256 does not match, file removed. Try again.')
    print(f'done: {len(names)} files in {args.out}/')


if __name__ == '__main__':
    main()
