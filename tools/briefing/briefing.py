"""Live project briefing for an outside auditor: JSON state in, one Markdown page out.

Why a script and not just editing the page by hand: a hand-edited status page rots. Items get
added and never closed, two entries contradict each other, and "what is outdated" becomes a
judgement call every time. Here the state is a small JSON file, the page is rendered from it,
and pruning is a rule rather than an opinion.

The reader is assumed to be a competent engineer from OUTSIDE this project. So: no unexplained
jargon, no internal shorthand, every claim carries the number that supports it, and anything
that was later found to be wrong stays visible under "corrected" rather than being deleted -
an auditor needs to see that the record self-corrects.

  briefing.py now     "<what is running right now>" [--detail "..."]
  briefing.py add     <id> "<title>" --why "..." [--section next|blocked]
  briefing.py done    <id> --finding "<the one-line result>" [--media path ...] [--number "..."]
  briefing.py block   <id> --needs "<what would unblock it>"
  briefing.py correct <id> --was "<what we said>" --now "<what is true>" --why "..."
  briefing.py drop    <id>
  briefing.py render                 # rewrite the page from state
"""
import argparse
import json
import os
from datetime import datetime, timezone, timedelta

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
STATE = f'{REPO}/docs/.briefing_state.json'
PAGE = f'{REPO}/docs/000.Real-time Brefing.md'
KST = timezone(timedelta(hours=9))
DONE_KEEP = 8          # completed items shown in full; older ones collapse to one line
DROP_AFTER_DAYS = 14   # collapsed items older than this disappear entirely


def now():
    return datetime.now(KST).strftime('%Y-%m-%d %H:%M')


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding='utf-8'))
    return {'now': None, 'items': {}, 'order': [], 'updated': None}


def save(s):
    s['updated'] = now()
    json.dump(s, open(STATE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    render(s)


def touch(s, tid, **kw):
    it = s['items'].setdefault(tid, {'id': tid, 'created': now()})
    it.update(kw)
    it['changed'] = now()
    if tid not in s['order']:
        s['order'].append(tid)
    return it


def dedupe_order(s):
    """Two sessions writing the state at once can append the same id twice, and prune()
    then deletes the item on the first copy and raises KeyError on the second. Cheap to
    make idempotent here rather than guard every reader."""
    seen, keep = set(), []
    for tid in s['order']:
        if tid in seen or tid not in s['items']:
            continue
        seen.add(tid)
        keep.append(tid)
    s['order'] = keep


def prune(s):
    """Collapse old completed work, drop what nobody will audit any more."""
    dedupe_order(s)
    done = [t for t in s['order'] if s['items'][t].get('section') == 'done']
    for tid in done[:-DONE_KEEP] if len(done) > DONE_KEEP else []:
        s['items'][tid]['collapsed'] = True
    cutoff = datetime.now(KST) - timedelta(days=DROP_AFTER_DAYS)
    for tid in list(s['order']):
        it = s['items'][tid]
        if it.get('collapsed') and it.get('changed'):
            try:
                if datetime.strptime(it['changed'], '%Y-%m-%d %H:%M').replace(tzinfo=KST) < cutoff:
                    s['order'].remove(tid)
                    s['items'].pop(tid)
            except ValueError:
                pass


def _media(it):
    out = []
    for p in it.get('media', []):
        rel = p.replace(f'{REPO}/docs/', '').replace(f'{REPO}/', '../')
        out.append(f'![[{os.path.basename(rel)}]]' if rel.endswith(('.png', '.mp4', '.jpg'))
                   else f'[{os.path.basename(rel)}]({rel})')
    return out


def render(s):
    L = [f'# 실시간 브리핑 — Pygmalion 보행 로봇',
         '',
         f'*마지막 갱신 {s.get("updated") or now()} (KST). 이 페이지는 자동 생성됩니다 — '
         '손으로 고치지 말고 `tools/briefing/briefing.py`로 바꾸세요.*',
         '',
         '> **이 프로젝트가 뭘 하는 건가**: 두 발 로봇을 시뮬레이션 안에서 걷도록 학습시키고, '
         '걸을 때 관절과 뼈대에 걸리는 **힘을 측정해서 실제 하드웨어를 어떤 크기로 만들지 정하는 것**이 목표입니다. '
         '로봇을 실제로 만들기 전에, 모터가 얼마나 세야 하는지·부품이 얼마나 튼튼해야 하는지를 숫자로 알아내려는 겁니다.',
         '']

    if s.get('now'):
        L += ['## 지금 하고 있는 일', '', f'**{s["now"]["title"]}**', '']
        if s['now'].get('detail'):
            L += [s['now']['detail'], '']
        L += [f'*시작 {s["now"]["since"]}*', '']

    groups = [('next', '## 다음에 할 일', '아직 시작하지 않았습니다.'),
              ('blocked', '## 막혀 있는 일', '없습니다.'),
              ('done', '## 끝난 일 (최신순)', '아직 없습니다.'),
              ('correct', '## 정정된 결론', '아직 없습니다.')]
    for sec, head, empty in groups:
        items = [s['items'][t] for t in s['order'] if s['items'][t].get('section') == sec]
        if sec == 'done':
            items = items[::-1]
        L += [head, '']
        if not items:
            L += [f'*{empty}*', '']
            continue
        for it in items:
            if it.get('collapsed'):
                L.append(f'- ~~{it["title"]}~~ — {it.get("finding", "")} *(요약 보관)*')
                continue
            L.append(f'### {it["title"]}')
            if it.get('why'):
                # in the corrections section "why" answers how the error surfaced, not why the
                # work was undertaken - same field, different question
                lbl = '어떻게 발견했나' if sec == 'correct' else '왜 하는가'
                L += ['', f'**{lbl}** — {it["why"]}']
            if it.get('finding'):
                L += ['', f'**결과** — {it["finding"]}']
            if it.get('number'):
                L += ['', f'**근거 숫자** — {it["number"]}']
            if it.get('needs'):
                L += ['', f'**풀리려면** — {it["needs"]}']
            if it.get('was'):
                L += ['', f'**처음 결론** — {it["was"]}', '', f'**바로잡은 결론** — {it["now_"]}']
            for m in _media(it):
                L += ['', m]
            L += ['', f'*{it.get("changed", "")}*', '']
        L.append('')
    open(PAGE, 'w', encoding='utf-8').write('\n'.join(L).rstrip() + '\n')
    print(f'rendered {PAGE}')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('now'); a.add_argument('title'); a.add_argument('--detail', default='')
    for name in ('add', 'done', 'block', 'correct', 'drop'):
        q = sub.add_parser(name)
        q.add_argument('id')
        q.add_argument('title', nargs='?', default=None)
        q.add_argument('--why', default=None)
        q.add_argument('--finding', default=None)
        q.add_argument('--number', default=None)
        q.add_argument('--needs', default=None)
        q.add_argument('--was', default=None)
        q.add_argument('--now', dest='now_', default=None)
        q.add_argument('--media', nargs='*', default=None)
        q.add_argument('--section', default=None)
    sub.add_parser('render')
    args = p.parse_args()
    s = load()

    if args.cmd == 'now':
        s['now'] = {'title': args.title, 'detail': args.detail, 'since': now()}
    elif args.cmd == 'render':
        pass
    elif args.cmd == 'drop':
        s['order'] = [t for t in s['order'] if t != args.id]
        s['items'].pop(args.id, None)
    else:
        kw = {k: v for k, v in vars(args).items()
              if v is not None and k not in ('cmd', 'id', 'section')}
        kw.pop('now_', None) if args.now_ is None else None
        if args.title:
            kw['title'] = args.title
        elif args.id in s['items']:
            kw['title'] = s['items'][args.id]['title']
        sec = args.section or {'add': 'next', 'done': 'done',
                               'block': 'blocked', 'correct': 'correct'}[args.cmd]
        kw['section'] = sec
        touch(s, args.id, **kw)
    prune(s)
    save(s)


if __name__ == '__main__':
    main()
