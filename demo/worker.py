import numpy
import functools
import os
import re
import math

from aiohttp import web
import tflite_runtime.interpreter as tflite

CTX_LEN = 32
MODEL = tflite.Interpreter(model_path="../model/lite/model.tflite")

# do not use caching in you real project because passwords can leak
@functools.lru_cache(maxsize=100)
def predict(text):
    text = " " + text
    tokens = [ord(c) for c in text[:CTX_LEN-1]]
    tokens += [0] * (CTX_LEN - len(tokens))

    f = MODEL.get_signature_runner("serving_default")
    p = f(input=numpy.array([tokens], dtype=numpy.int32))["outputs"][0][0]
    p = numpy.exp(p) / numpy.sum(numpy.exp(p))
    return p


def check_password(password, brute_speed=1_000_000, good_brute_duration=86400*30):
    print(password)
    password = re.sub(r"[^\x20-\x7e]", ".", password)

    probs = {}
    for pos in range(len(password)+1):
        subword = password[:pos]
        next_char_probs = predict(subword)
        next_char_probs[0] = 0.0
        next_char_probs /= numpy.sum(next_char_probs)

        p = {}
        for char in sorted(range(len(next_char_probs)), key=lambda k: -next_char_probs[k]):
            p[chr(char)] = float(next_char_probs[char])

        probs[subword] = p

    # anti-bruteforce probability clipping
    variants = 95  # ascii
    if not re.search(r"[a-z]", password):
        variants -= 26
    if not re.search(r"[A-Z]", password):
        variants -= 26
    if not re.search(r"[0-9]", password):
        variants -= 10
    if not re.search(r"[^0-9A-Za-z]", password):
        variants -= 95 - 10 - 26 - 26
    if variants == 0:
        variants = 95

    prob = 1.0
    uncutted_prob = 1.0

    for pos in range(len(password)):
        part = password[:pos]
        cur_prob = probs[part][password[pos]]

        uncutted_prob *= cur_prob

        if cur_prob <= (1.0 / variants):
            cur_prob = 1.0 / variants

        prob *= cur_prob

    secs = 1/prob/brute_speed

    complexity = math.log(1/uncutted_prob) / math.log(95)

    percent = ((math.log2(secs)/math.log2(64))/(math.log2(good_brute_duration)/math.log2(64)))*100
    percent = max(percent, 2*len(password) if len(password) < 20 else 40+(len(password)-20))
    percent = min(percent, 100)

    return {"secs": secs, "percent": percent, "complexity": complexity, "probs": probs}


async def web_check_password(request, brute_speed=1_000_000, good_brute_duration=86400*30):
    password = request.query.get('password', "")[:CTX_LEN]
    return web.json_response(check_password(password, brute_speed, good_brute_duration))

if __name__ == "__main__":
    app = web.Application()
    app.add_routes([
        web.get('/', lambda r: web.FileResponse("index.html")),
        web.get('/magic', lambda r: web.FileResponse("inbrowser.html")),
        web.get('/model.json', lambda r: web.FileResponse("../model/web/model.json")),
        web.get('/check_password', web_check_password)])
    for f in os.listdir("../model/web"):
        if re.fullmatch(r"group\d+-shard\d+of\d+\.bin", f):
            app.add_routes([web.get("/" + f, lambda r, f2=f: web.FileResponse("../model/web/"+f2))])

    web.run_app(app, port=80)
