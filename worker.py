import numpy
import functools
import sys
import os
import re

from aiohttp import web
import tensorflow as tf

CTX_LEN = 32
MODEL = tf.saved_model.load("model")

@functools.lru_cache(maxsize=10000)
def predict(text):
    text = " " + text
    tokens = [ord(c) for c in text[:CTX_LEN-1]]
    tokens += [0] * (CTX_LEN - len(tokens))

    t = tf.convert_to_tensor([tokens], dtype=numpy.int32)
    p = MODEL.signatures["serving_default"](input=t)["outputs"][0][0].numpy()
    p = numpy.exp(p) / numpy.sum(numpy.exp(p))
    p[0] = 0.0
    p /= numpy.sum(p)

    ans = {}
    for pos in sorted(range(len(p)), key=lambda k: -p[k]):
        ans[chr(pos)] = float(p[pos])
    return ans


async def check_password(request):
    password = request.query.get('password', "")[:CTX_LEN]
    ans = {}
    for pos in range(len(password)+1):
        subword = password[:pos]
        ans[subword] = predict(subword)
    return web.json_response(ans)


if __name__ == "__main__":
    app = web.Application()
    app.add_routes([
        web.get('/', lambda r: web.FileResponse("index.html")),
        web.get('/magic', lambda r: web.FileResponse("inbrowser.html")),
        web.get('/model.json', lambda r: web.FileResponse("web_model/model.json")),
        web.get('/check_password', check_password)])
    for f in os.listdir("web_model"):
        if re.fullmatch(r"group\d+-shard\d+of\d+\.bin", f):
            app.add_routes([web.get("/" + f, lambda r, f2=f: web.FileResponse("web_model/"+f2))])

    web.run_app(app, port=80)
