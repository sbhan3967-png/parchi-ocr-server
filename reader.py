"""Parchi Reader core: OCR + tilt fix + table parse + bifurcation logic."""
import re
import math
import numpy as np

# ---------------------------------------------------------------- OCR
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def ocr_words(img_bgr):
    """Return list of dicts: t, c (0-100), x0,y0,x1,y1, xc, yc.
    RapidOCR returns a list of [4-point box, text, score]."""
    eng = get_engine()
    result, _ = eng(img_bgr)
    words = []
    if result:
        for entry in result:
            box, txt, sc = entry[0], entry[1], entry[2]
            if not txt or not txt.strip():
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            words.append(dict(t=txt.strip(), c=float(sc) * 100,
                              x0=x0, y0=y0, x1=x1, y1=y1,
                              xc=(x0 + x1) / 2, yc=(y0 + y1) / 2))
    return words


# ------------------------------------------------- tilt estimation
def estimate_tilt(words, W, H):
    """Projection-profile search: the rotation giving the peakiest y-histogram."""
    if len(words) < 20:
        return 0.0
    pts = np.array([(w['xc'], w['yc']) for w in words])
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-4.0, 4.01, 0.5):
        t = math.radians(a)
        ys = (pts[:, 1] - H / 2) * math.cos(t) - (pts[:, 0] - W / 2) * math.sin(t)
        hist, _ = np.histogram(ys, bins=max(10, int(H / 8)))
        s = float(np.var(hist))
        if s > best_s:
            best_s, best_a = s, a
    return best_a


# ------------------------------------------------- helpers
def clean_num(t):
    t = t.replace(',', '.')
    t = re.sub(r'[^0-9.]', '', t)
    t = re.sub(r'\.+', '.', t)
    return t


def is_num(t):
    return bool(re.match(r'^\d+\.\d+$', clean_num(t)))


ZONES = {'gwt': (0.28, 0.375), 'less': (0.375, 0.46), 'net': (0.46, 0.56),
         'tw': (0.56, 0.62), 'labour': (0.62, 0.70), 'silver': (0.70, 0.83),
         'amount': (0.83, 1.01)}


def zone_of(x, W):
    f = x / W
    for k, (a, b) in ZONES.items():
        if a <= f < b:
            return k
    return 'item' if f < 0.28 else 'amount'


def is_labour_token(t):
    """Labour rates/totals, not weights: 300.00Wt, 25000.00WI, {8.00x55.00=440.00)."""
    return bool(re.search(r'W(T|I)?\b', t, re.I) or re.search(r'\d\s*[x\u00d7%]\s*\d', t))


_wref = []


def find_triple(cands, W):
    """Find (net, tw, silver) using the invariant silver == net * tw / 100.
    cands: list of (value, xc, conf, word). Returns list of same with word refs."""
    global _wref
    _wref = [{'v': v, 'x': x, 'w': w} for v, x, c, w in cands]
    best = None
    for nv, nx, nc, *_ in cands:
        if not (0.5 <= nv <= 50000):
            continue
        for tv, tx, tc, *_ in cands:
            if not (20 <= tv <= 200):
                continue
            dx1 = tx - nx
            if not (W * 0.005 < dx1 < W * 0.15):
                continue
            for sv, sx, sc, *_ in cands:
                dx2 = sx - tx
                if not (W * 0.03 < dx2 < W * 0.35):
                    continue
                if abs(nv * tv / 100 - sv) > max(0.02, sv * 0.001):
                    continue
                score = nc + tc + sc
                if 0.45 < nx / W < 0.60:
                    score += 0.5
                if best is None or score > best[0]:
                    best = (score, (nv, nx, nc), (tv, tx, tc), (sv, sx, sc))
    if best is None:
        return None
    # re-attach word refs
    out = []
    for v, x, c in best[1:]:
        w = next(d['w'] for d in _wref if abs(d['x'] - x) < 0.01 and abs(d['v'] - v) < 1e-9)
        out.append((v, x, c, w))
    return out


# ------------------------------------------------- table parse
def parse_words(words, W, H):
    tilt = estimate_tilt(words, W, H)
    tn = math.tan(math.radians(tilt))
    for w in words:
        w['yc'] = w['yc'] - w['xc'] * tn
    words = sorted(words, key=lambda w: w['yc'])
    TOL = H * 0.014
    rows = []
    for w in words:
        r = next((r for r in rows if abs(r['yc'] - w['yc']) < TOL), None)
        if r is None:
            r = {'yc': w['yc'], 'words': []}
            rows.append(r)
        r['words'].append(w)
        r['yc'] = sum(x['yc'] for x in r['words']) / len(r['words'])
    rows.sort(key=lambda r: r['yc'])
    for r in rows:
        r['words'].sort(key=lambda w: w['x0'])

    SNO_MAX = 50  # bigger "S-numbers" are tunch/model remnants, not serials
    items, cur = [], None
    for r in rows:
        left_w = [w for w in r['words'] if w['xc'] / W < 0.32]
        left_text = ' '.join(w['t'] for w in left_w)
        m = re.search(r'\bS\D{0,2}(\d{1,2})(?!\d)', left_text, re.I)
        sno = int(m.group(1)) if m and int(m.group(1)) <= SNO_MAX else None
        if sno is None:
            mb = re.match(r'(\d{1,2})(?=\s|[A-Z(\uff08])', left_text)
            if mb and int(mb.group(1)) <= SNO_MAX:
                sno = int(mb.group(1))
        nums = [w for w in r['words'] if is_num(w['t'])]
        line_text = ' '.join(w['t'] for w in r['words'])
        is_packet = (('(' in left_text) or ('\uff08' in left_text)) and sno is None
        is_piece = bool(re.search(r'\d\s*[x\u00d7%]\s*\d[\d.]*\s*=', line_text))
        has_alpha = any(re.search(r'[A-Za-z]{2,}', w['t']) for w in left_w)
        cur_nums = sum(1 for rr in (cur['rows'] if cur else [])
                       for w in rr['words'] if is_num(w['t']))
        data_like = len(nums) >= 3 and has_alpha
        cur_complete = cur_nums >= 2
        if sno is not None:
            cur = {'sno': sno, 'rows': [r]}
            items.append(cur)
        elif data_like:
            if cur is None or cur_complete:
                cur = {'sno': 0, 'rows': [r]}
                items.append(cur)
            else:
                cur['rows'].append(r)          # rest of the same item's numbers
        elif cur is None:
            pass                              # ignore stray line
        elif is_piece and abs(r['yc'] - cur['rows'][-1]['yc']) < H * 0.06:
            cur['rows'].append(r)
        elif is_packet or (has_alpha and not nums):
            cur['rows'].append(r)
        elif len(nums) >= 2 and not cur_complete:
            cur['rows'].append(r)
        # else: ignore stray line

    # fill missing S-numbers: backward then forward from known anchors, else sequential
    for i in range(len(items) - 2, -1, -1):
        if items[i]['sno'] == 0 and items[i + 1]['sno'] != 0:
            items[i]['sno'] = items[i + 1]['sno'] - 1
    for i in range(1, len(items)):
        if items[i]['sno'] == 0 and items[i - 1]['sno'] != 0:
            items[i]['sno'] = items[i - 1]['sno'] + 1
    for i, it in enumerate(items):
        if it['sno'] == 0:
            it['sno'] = i + 1

    out = []
    for it in items:
        allw = [w for r in it['rows'] for w in r['words']]
        numws = [(float(clean_num(w['t'])), w['xc'], w['c'], w['t'])
                 for w in allw if is_num(w['t'])]
        numws = [{'v': float(clean_num(w['t'])), 'x': w['xc'], 'c': w['c'],
                      't': w['t'], 'w': w} for w in allw if is_num(w['t'])]
        triple_cands = [(d['v'], d['x'], d['c'], d['w'])
                        for d in numws if not is_labour_token(d['t'])]
        net = tw = silver = ''
        silver_w = None
        gwt = ''
        hit = find_triple(triple_cands, W)
        if hit:
            (nv, nx, _nc, nw), (tv, tx, _tc, _tw2), (sv, sx, _sc, sw) = hit
            net, tw, silver, silver_w = f'{nv:g}', f'{tv:g}', f'{sv:g}', sw
            # G.Wt: rightmost number left of Net with value > net (gross > net);
            # skips tunch-in-name and Less column. Sanity: < 3x net.
            cands = [(d['v'], d['x']) for d in numws
                     if d['x'] < nx - 1 and not is_labour_token(d['t']) and d['v'] > nv]
            cands.sort(key=lambda p: -p[1])
            for v, _x in cands:
                if v < nv * 3:
                    gwt = f'{v:g}'
                    break
        if not net:
            cols = {k: [] for k in ZONES}
            for d in numws:
                z = zone_of(d['x'], W)
                if z in cols:
                    cols[z].append((f"{d['v']:g}", d['c']))

            def pick(k):
                a = cols[k]
                if not a:
                    return ''
                a.sort(key=lambda x: -x[1])
                return a[0][0]

            net, tw, silver = pick('net'), pick('tw'), pick('silver')
            gwt = pick('gwt')

        item = ' '.join(w['t'] for w in it['rows'][0]['words'] if w['xc'] / W < 0.34)
        item = re.sub(r'^.*?\bS\D{0,2}\d{1,2}(?!\d)\s*', '', item, flags=re.I)
        item = re.sub(r'^\d{1,2}\s+', '', item)
        item = re.sub(r'[(\uff08].*?[)\uff09]', '', item)
        # prettify glued tokens: AV91PAYAL -> AV 91 PAYAL, BARTAN92.5 -> BARTAN 92.5
        item = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', item)
        item = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', item)
        item = re.sub(r'\s+', ' ', item).strip()
        # right-side text (labour patterns live here; packets live left of 0.5W)
        rtext = ' '.join(w['t'] for w in allw if w['xc'] / W > 0.5)
        amt_cands = [d for d in numws
                     if zone_of(d['x'], W) in ('silver', 'amount')
                     and not is_labour_token(d['t'])
                     and d['w'] is not silver_w]
        amt_text = f"{max(amt_cands, key=lambda d: d['x'])['v']:g}" if amt_cands else ''
        out.append({'sno': it['sno'], 'item': item, 'gwt': gwt,
                    'net': net, 'tw': tw, 'silver': silver,
                    'rtext': rtext, 'amt_text': amt_text,
                    'tunch': '', 'labour': '', 'flag': ''})
    return out, tilt


# ------------------------------------------------- business logic
def num(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def infer_tunch(item, tw):
    nums = [float(m.group(1)) for m in re.finditer(r'(\d+(?:\.\d+)?)', item or '')]
    cand = next((v for v in reversed(nums) if 30 <= v <= 100), None)
    if cand is None:
        return tw, ''
    if tw is None:
        return cand, ''
    if cand <= tw:
        return cand, ''
    return tw, 'number in name exceeds T+W \u2014 model no.? tunch taken as T+W'


def parse_labour(rtext, amt_text, net):
    blob = (rtext or '') + ' ' + (amt_text or '')
    pm = re.search(r'(\d+(?:\.\d+)?)\s*[x\u00d7%]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', blob)
    amt = num(amt_text)
    wm = re.search(r'(\d+(?:\.\d+)?)\s*W(?:T|I)?\b', rtext or '', re.I)
    if pm:
        tot = float(pm.group(3))
        flag = ''
        if amt is not None and abs(amt - tot) > max(1, tot * 0.02):
            flag = f'Amount {amt:g} \u2260 piece total {tot:g}'
        return tot, flag
    if wm:
        if amt is not None:
            return amt, ''
        if net:
            return float(wm.group(1)) * net / 1000, ''
        return 0.0, ''
    if amt is not None:
        return 0.0, f'Amount {amt:g} ignored (no labour rate on parchi)'
    return 0.0, ''


def compute_row(r, rate_kg=205000):
    rate_g = rate_kg / 1000.0
    net, tw = num(r.get('net')), num(r.get('tw'))
    tunch = num(r.get('tunch'))
    flag = r.get('flag') or ''
    if tunch is None:
        tunch, f2 = infer_tunch(r.get('item') or '', tw)
        if f2:
            flag = (flag + ' | ' + f2) if flag else f2
    if tunch is not None and tunch > 100:
        flag = (flag + ' | ' if flag else '') + 'tunch > 100 impossible \u2014 confirm'
    labour = num(r.get('labour'))
    if labour is None:
        labour, f3 = parse_labour(r.get('rtext') or '', r.get('amt_text') or '', net)
        if f3:
            flag = (flag + ' | ' + f3) if flag else f3
    silver = net * tw / 100 if (net is not None and tw is not None) else None
    w_pct = (tw - tunch) if (tw is not None and tunch is not None) else None
    if w_pct is not None:
        w_rsg = w_pct / 100 * rate_g
    elif tw is None:
        w_rsg = 0.0  # no T+W on parchi -> labour-only deal
    else:
        w_rsg = None
    l_rsg = labour / net if (labour is not None and net) else None
    # ---- user overrides from the verify table (empty cell = auto) ----
    ov = r.get('_ov') or {}
    if 'w_rsg' in ov:
        w_rsg = ov['w_rsg']
        w_pct = w_rsg / rate_g * 100 if rate_g else None
    elif 'w_pct' in ov:
        w_pct = ov['w_pct']
        w_rsg = w_pct / 100 * rate_g if rate_g else None
    if 'l_rsg' in ov:
        l_rsg = ov['l_rsg']
        if net:
            labour = l_rsg * net
    total = (w_rsg + l_rsg) if (w_rsg is not None and l_rsg is not None) else None
    metal_val = silver * rate_g if silver is not None else None
    total_amt = (metal_val + (labour or 0.0)) if metal_val is not None else None
    return dict(r, tunch=tunch, labour=labour, silver=silver,
                w_pct=w_pct, w_rsg=w_rsg, l_rsg=l_rsg, total=total,
                metal_val=metal_val, total_amt=total_amt, flag=flag)


# ------------------------------------------------- high-level
def read_image(path_or_array):
    import cv2
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array)
    else:
        img = path_or_array
    H, W = img.shape[:2]
    words = ocr_words(img)
    rows, tilt = parse_words(words, W, H)
    return rows, tilt, (W, H)


def read_pdf(path, dpi=200):
    """Render each PDF page to an image array (BGR, for cv2/RapidOCR)."""
    import fitz
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            arr = arr[:, :, :3][:, :, ::-1]  # RGBA -> BGR
        else:
            arr = arr[:, :, ::-1]  # RGB -> BGR
        pages.append((f'page {i + 1}', arr))
    doc.close()
    return pages
