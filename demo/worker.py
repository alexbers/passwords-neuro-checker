import os
import re
import functools

from aiohttp import web

import neuro_pass_checker

async def web_check_password(request, brute_speed=1_000_000, good_brute_duration=86400*30):
    password = request.query.get('password', "")
    return web.json_response(neuro_pass_checker.check_password(password, brute_speed, good_brute_duration))

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
