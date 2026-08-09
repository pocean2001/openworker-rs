# -*- coding: utf-8 -*-
r"""
openworker-rs GitHub 一键同步脚本
=================================
通过 GitHub Contents API 将本地项目文件增量同步到远程仓库。
（参考 D:\hf-flux\sync_github.py 的机制：绕过本机不稳定的 git 协议，api.github.com 可正常访问）

用法:
    python sync_github.py --token-file D:/hf-flux/token.txt
    python sync_github.py --token ghp_xxx        # 直接传 token
    python sync_github.py --token-file D:/hf-flux/token.txt --dry-run

说明:
    - 同步文件清单由 `git ls-files` 动态生成（自动遵循 .gitignore，
      因而 openworker.local.toml / target/ / archive/ 等机密与产物不会上传）。
    - 上传采用 SHA 比对：本地未变则跳过，已变则更新，远程缺失则新建。
    - token 默认从同目录 token.txt 读取；也可用 --token / --token-file 显式传入，
      切勿把真实 token 提交进仓库。
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error

# ============================================================================
# 配置（可用命令行参数覆盖）
# ============================================================================
REPO = 'pocean2001/openworker-rs'
BRANCH = 'main'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(BASE_DIR, 'token.txt')


def get_token(arg_token, arg_token_file):
    """获取 token：命令行 --token > --token-file > 同目录 token.txt。"""
    if arg_token:
        return arg_token.strip()
    if arg_token_file and os.path.isfile(arg_token_file):
        with open(arg_token_file, 'r', encoding='utf-8') as f:
            t = f.read().strip()
        if t:
            return t
    if os.path.isfile(TOKEN_FILE):
        with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
            t = f.read().strip()
        if t:
            return t
    print('❌ 未找到 token！请用 --token 或 --token-file 传入 GitHub Personal Access Token。')
    print('   创建方法: https://github.com/settings/tokens → Generate new token (勾选 repo)')
    sys.exit(1)


def list_sync_files():
    """用 git ls-files 动态获取应上传的文件（已遵循 .gitignore）。"""
    try:
        out = subprocess.check_output(
            ['git', 'ls-files'], cwd=BASE_DIR, encoding='utf-8')
        files = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if files:
            return files
    except Exception as e:
        print(f'  ⚠ git ls-files 失败: {e}')
    return []


def api_call(path, method='GET', data=None, token=None):
    """调用 GitHub API。"""
    req = urllib.request.Request(
        f'https://api.github.com{path}',
        method=method,
        headers={
            'Authorization': f'token {token}',
            'User-Agent': 'openworker-rs-sync',
            'Content-Type': 'application/json',
        })
    if data is not None:
        req.data = json.dumps(data).encode('utf-8')
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        body = resp.read()
        return json.loads(body) if body else {}, None
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        return None, f'HTTP {e.code}: {body[:200]}'


def get_remote_sha(path, token):
    """获取远程文件的 SHA（不存在返回 None）。"""
    r, err = api_call(f'/repos/{REPO}/contents/{path}?ref={BRANCH}', token=token)
    if err:
        if '404' in err:
            return None
        print(f'  ⚠ 查询 {path} SHA 失败: {err}')
        return None
    return r.get('sha')


def sync_file(path, token, dry_run=False):
    """同步单个文件。返回 (状态, 消息)。"""
    local_path = os.path.join(BASE_DIR, path)
    if not os.path.isfile(local_path):
        return 'skip', f'{path} 本地不存在，跳过'

    with open(local_path, 'rb') as f:
        content = f.read()
    local_sha = hashlib.sha256(content).hexdigest()

    remote_sha = get_remote_sha(path, token)

    # 无远程文件 → 新建
    if remote_sha is None:
        if dry_run:
            return 'new', f'{path} 将新建'
        data = {
            'message': f'add {path}',
            'content': base64.b64encode(content).decode('utf-8'),
            'branch': BRANCH,
        }
        r, err = api_call(f'/repos/{REPO}/contents/{path}', 'PUT', data, token)
        if err:
            return 'fail', f'{path} 新建失败: {err}'
        return 'new', f'{path} 新建成功'

    # 有远程文件 → 下载对比内容
    r, err = api_call(f'/repos/{REPO}/contents/{path}?ref={BRANCH}', token=token)
    if err:
        return 'fail', f'{path} 读取远程失败: {err}'
    try:
        remote_content = base64.b64decode(r['content']).decode('utf-8', errors='replace')
        remote_sha_cmp = hashlib.sha256(remote_content.encode('utf-8', errors='replace')).hexdigest()
    except Exception:
        remote_sha_cmp = None

    if remote_sha_cmp == local_sha:
        return 'same', f'{path} 无变化'

    if dry_run:
        return 'diff', f'{path} 有差异，将更新'
    data = {
        'message': f'update {path}',
        'content': base64.b64encode(content).decode('utf-8'),
        'sha': remote_sha,
        'branch': BRANCH,
    }
    r, err = api_call(f'/repos/{REPO}/contents/{path}', 'PUT', data, token)
    if err:
        return 'fail', f'{path} 更新失败: {err}'
    return 'diff', f'{path} 更新成功'


def main():
    global REPO, BRANCH
    parser = argparse.ArgumentParser(description='openworker-rs GitHub 一键同步')
    parser.add_argument('--token', default=None, help='GitHub token（明文）')
    parser.add_argument('--token-file', default=None, help='存放 GitHub token 的文件路径')
    parser.add_argument('--repo', default=REPO, help=f'目标仓库 (默认 {REPO})')
    parser.add_argument('--branch', default=BRANCH, help=f'目标分支 (默认 {BRANCH})')
    parser.add_argument('--dry-run', action='store_true', help='只显示差异不上传')
    args = parser.parse_args()

    REPO = args.repo
    BRANCH = args.branch

    token = get_token(args.token, args.token_file)
    files = list_sync_files()
    print(f'仓库: {REPO}  分支: {BRANCH}')
    print(f'模式: {"DRY-RUN (不实际上传)" if args.dry_run else "同步"}')
    print(f'待同步文件数: {len(files)}\n')

    results = {'new': [], 'diff': [], 'same': [], 'fail': [], 'skip': []}
    for f in files:
        status, msg = sync_file(f, token, args.dry_run)
        results[status].append(msg)
        icon = {'new': '🆕', 'diff': '🔄', 'same': '✅', 'fail': '❌', 'skip': '⏭'}[status]
        print(f'  {icon} {msg}')

    print('\n' + '=' * 50)
    print(f"新建: {len(results['new'])}  更新: {len(results['diff'])}  "
          f"无变化: {len(results['same'])}  失败: {len(results['fail'])}  跳过: {len(results['skip'])}")
    if results['fail']:
        print('\n⚠️ 以下文件同步失败:')
        for m in results['fail']:
            print(f'  - {m}')
    print(f'\n完成! 访问 https://github.com/{REPO} 查看')


if __name__ == '__main__':
    main()
