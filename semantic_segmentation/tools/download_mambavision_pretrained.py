#!/usr/bin/env python3
"""将 MambaVision-Tiny-1K 预训练权重下载到 checkpoint/pretrained/（与 wire 配置默认路径一致）。"""
import argparse
import os
import shutil
import sys
import urllib.request

DEFAULT_URL = (
    'https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar')
DEFAULT_REL = os.path.join('checkpoint', 'pretrained', 'mambavision_tiny_1k.pth.tar')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        default=DEFAULT_REL,
        help='输出路径（相对 cwd 或绝对路径），默认 checkpoint/pretrained/mambavision_tiny_1k.pth.tar')
    parser.add_argument('--url', default=DEFAULT_URL, help='下载地址')
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='已存在文件时仍重新下载')
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.isfile(out) and not args.force:
        print(f'已存在，跳过: {out}', file=sys.stderr)
        return 0

    tmp = out + '.partial'
    print(f'下载 -> {tmp}', file=sys.stderr)
    try:
        urllib.request.urlretrieve(args.url, tmp)
    except Exception as e:
        if os.path.isfile(tmp):
            os.remove(tmp)
        print(f'下载失败: {e}', file=sys.stderr)
        return 1
    shutil.move(tmp, out)
    print(f'完成: {out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
